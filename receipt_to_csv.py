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
        dict or list: The parsed data from the YAML file
                        Returns None if the file cannot be read or parsed.
    """
    try:
        with open(file_path, 'r') as file:
            # Use yaml.safe_load for security
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        exit(f"Error: The file '{file_path}' was not found.")
    except yaml.YAMLError as e:
        exit(f"Error parsing YAML file '{file_path}': {e}")

def encode_image(image_path: str) -> str:
    """
    Encodes a localy image document to base64
    Args:
        image_path (str): The path to the image file.
    Returns:
        string of the encoded image
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def create_prompt_message(prompt_list: list, image_base64: str) -> None:
    """
    Given an image, and the list  creates a ChatPromptTemplate object
    and appends it to a list
    Args:
        prompt_list (list): The list of all prompts with encoded images
        image_base64 (str): The image encoded data
    Returns:
        None, appends to the list
    """
    prompt_messages = [
        SystemMessage(content="This is a shopping receipt"),
        HumanMessagePromptTemplate.from_template(
            template=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": "{text_prompt}"}
            ]
        ),
    ]
    prompt_list.append(ChatPromptTemplate(messages=prompt_messages))

# Setup for extensability with other llm models.
def build_genai(my_class: type, **kwargs):
    """
    wrapper set to handle different langchain llm classes
    Args:
        class (class): The GenAI specific class to use
        **kwargs (dict): The corresponding parameters for the GenAI class
    Returns:
        The object to interact with the desired langchain LLM model
    """
    try:
        llm = my_class(**kwargs)
    except Exception as e:
        # catch all errors and exit
        # This can de developped later
        exit(f"An unhandled error occurred: {e}")
    return llm

def call_llm(prompt: type, llm: type) -> str:
    """
    Function used to make a prompt to the LLM
    Args:
        prompt (object): The pre formated prompt template to improve outputs
        llm (object): The object used to handle calls to the LLM
    Returns:
        A string representing a csv of the receipt.
    """

    # setting the
    chain = prompt | llm

    # Invoke the chain with the image and text prompt
    response = chain.invoke({"text_prompt": """Make a table out of the shopping receipt
                            with these columns: shop_name, shop_address, date,
                            item_name, item_count, item_cost, cart_tax,
                            cart_total.
                            If information is not available fill with null.
                            The values of the following should be equal:
                            shop_name, shop_address, date, cart_tax, cart_total
                            Return as csv with delimiter ;"""})
    # Formatting based on observed

    return clean_response(response)

def clean_response(response: str) -> str:
    """
    Used to clean up the LLM response for ease of processing
    Args:
        response (string): The GenAI specific class to use
        **kwargs (dict): The corresponding parameters for the GenAI class
    Returns:
        The object to interact with the desired langchain LLM model
    """
    res = response.content
    res = res.replace("`", '')
    res = res.replace("csv", '')
    return res

def save_as_csv(response: str, df_out: type[pd.DataFrame]) -> None:
    # converting string into stream for conversion to data frame
    csv_string = io.StringIO(response)

    df = pd.read_csv(csv_string, delimiter=';')
    df = df[['shop_name', 'shop_address', 'date', 'item_name', 'item_count', 'item_cost', 'cart_tax', 'cart_total']]
    df_out = pd.concat([df_out, df])

def build_receipts_table():
    # Retreivin the credentials and configuration
    creds = read_yaml_file('creds.yaml')
    config = read_yaml_file('config.yaml')
    # getting the list of images to process
    images_folder = config['paths']['images_location']
    image_paths = os.listdir(images_folder)
    # Creating the list to hold all image data promts
    prompts = []

    # Prepare the Image Data (from local file)
    for image in image_paths:
        image_base64 = encode_image(images_folder+image)

        # Construct the Prompts formats
        create_prompt_message(prompts, image_base64)

    # Setting up the LLM
    llm = build_genai(
        my_class=ChatGoogleGenerativeAI,
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=creds['keys']['gemini']
    )

    # creating an empty data frame to hold
    df_out = pd.DataFrame()

    # Create a chain
    for prompt in prompts:

        res = call_llm(prompt, llm)
        save_as_csv(res, df_out)


    df_out.to_csv('receipts.csv', index = False)

    print('Done')

if __name__ == '__main__':
    build_receipts_table()
