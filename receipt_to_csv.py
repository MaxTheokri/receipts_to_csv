import base64
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage  # <-- fixed import
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd 
import io
import os 
import yaml

# retreiving the creds
def read_yaml_file(file_path: str) -> str:
    """
    Reads and parses a YAML file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict or list: The parsed data from the YAML file, typically a dictionary
                        or a list, depending on the YAML structure.
                        Returns None if the file cannot be read or parsed.
    """
    try:
        with open(file_path, 'r') as file:
            # Use yaml.safe_load for security when dealing with untrusted YAML sources.
            # yaml.FullLoader can be used if the source is trusted.
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        exit(f"Error: The file '{file_path}' was not found.")

    except yaml.YAMLError as e:
        exit(f"Error parsing YAML file '{file_path}': {e}")



creds = read_yaml_file('creds.yaml')
config = read_yaml_file('config.yaml')


# 1. Prepare the Image Data (from local file)
def encode_image(image_path: str) -> str:
    """
    encodes a localy image document to base64 

    Args:
        image_path (str): The path to the image file.

    Returns:
        string of the encoded image
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    images_folder = config['paths']['images_location']
    image_paths = os.listdir(images_folder)
    prompts = []

    for image in image_paths:
        image_base64 = encode_image(images_folder+image)

        # 2. Construct the Multimodal Prompt
        prompt_messages = [
            SystemMessage(content="This is a shopping receipt"),
            HumanMessagePromptTemplate.from_template(
                template=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "{text_prompt}"}
                ]
            ),
        ]
        prompts.append(ChatPromptTemplate(messages=prompt_messages))

    # 3. Define and Invoke the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=creds['keys']['gemini']
    )

    df_out = pd.DataFrame()

    # Create a chain
    for prompt in prompts:
        chain = prompt | llm

        # Invoke the chain with the image and text prompt
        response = chain.invoke({"text_prompt": "Make a table out of it with these columns: shop_name, shop_address, date, item_name, item_count, item_cost, cart_tax, cart_total. If information is not available fill with null. Return as csv with delimiters ;"})

        # Formatting
        res = response.content
        res = res.replace("`", '')
        res = res.replace("csv", '') 

        # converting string into stream for conversion to data frame
        csv_string = io.StringIO(res)

        df = pd.read_csv(csv_string, delimiter=';')
        df = df[['shop_name', 'shop_address', 'date', 'item_name', 'item_count', 'item_cost', 'cart_tax', 'cart_total']]
        df_out = pd.concat([df_out, df])

    df_out.to_csv('receipts.csv', index = False)

    print('Done')

if __name__ == '__main__':
    main()
