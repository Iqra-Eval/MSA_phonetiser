import sys
import json
import wave # This module was present in your original code, but not directly used in the phonetisation logic.
import string
import argparse
from phonetiser.phonetise_Arabic import phonetise # Assumes 'phonetiser' is installed and available
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd # Used for creating Dataset from CSV/text and saving to CSV

def remove_digits(text):
    """
    Removes all digits from the given text.

    Parameters:
        text (str): The input text.

    Returns:
        str: The text without digits.
    """
    if not isinstance(text, str):
        return text # Return as is if not a string (e.g., None or NaN)
    return ''.join(char for char in text if not char.isdigit())

def remove_punctuation(text):
    """
    Remove common punctuation from the input text.
    The original function specifically targeted "?", "!", and ".",
    but this uses `string.punctuation` for broader removal.

    Args:
        text (str): The input text.

    Returns:
        str: The text with specified punctuation removed.
    """
    if not isinstance(text, str):
        return text # Return as is if not a string
    # Create a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def get_phoneme(batch):
    """
    Processes a batch of text to generate phoneme sequences.
    This function is designed to be used with `dataset.map()`.

    Args:
        batch (dict): A dictionary containing a batch of data,
                      expected to have a 'text_to_phonetise' key.

    Returns:
        dict: The batch dictionary with a new 'phoneme_sequence' key.
    """
    texts = batch['text_to_phonetise']
    phoneme_sequences = []
    for text in texts:
        if text is None:
            phoneme_sequences.append(None) # Handle None values in text column
            continue

        # Clean text
        cleaned_text = remove_punctuation(text)
        cleaned_text = remove_digits(cleaned_text)

        # Phonetise Arabic text
        # phonetise() returns (original_text, list_of_phonemes_as_strings)
        try:
            # We are interested in the list of phonemes (index 1)
            # The original code joined with " " and replaced "sil" with "<sil>"
            phonemes = " ".join(phonetise(cleaned_text)[1]).replace("sil", "<sil>")
            # Original code added "<sil> " prefix. Let's maintain that.
            transcription = "<sil> " + phonemes
            phoneme_sequences.append(transcription)
        except Exception as e:
            print(f"Error phonetising text: '{text}'. Error: {e}", file=sys.stderr)
            phoneme_sequences.append(f"[ERROR_PHONETISING: {e}]") # Indicate error in output

    batch['phoneme_sequence'] = phoneme_sequences
    return batch

def main():
    parser = argparse.ArgumentParser(description="Phonetise Arabic text from various sources and output in different formats.")
    parser.add_argument("--input_type", type=str, required=True, choices=['huggingface', 'csv', 'text'],
                        help="Type of input data: 'huggingface' for a dataset name, 'csv' for a CSV file, 'text' for a plain text file.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input file (for 'csv' or 'text') or the Hugging Face dataset name (for 'huggingface').")
    parser.add_argument("--output_type", type=str, required=True, choices=['csv', 'text', 'huggingface'],
                        help="Type of output: 'csv' for a CSV file, 'text' for a plain text file, 'huggingface' for pushing to Hugging Face Hub.")
    parser.add_argument("--output_path", type=str,
                        help="Path to the output file (required for 'csv' or 'text' output_type).")
    parser.add_argument("--column_id", type=str, default="ID",
                        help="Column name for the ID in CSV input (default: 'ID').")
    parser.add_argument("--column_text", type=str, default="Arabic text",
                        help="Column name for the Arabic text in CSV input (default: 'Arabic text').")
    parser.add_argument("--hf_text_column", type=str, default=None,
                        help="Specific column name for Arabic text in Hugging Face datasets. If not provided, the script will try to infer it.")
    parser.add_argument("--hf_output_repo_id", type=str, default=None,
                        help="Hugging Face repository ID for the output dataset (required for 'huggingface' output_type).")


    args = parser.parse_args()

    # Validate output_path and hf_output_repo_id based on output_type
    if args.output_type in ['csv', 'text'] and not args.output_path:
        parser.error("--output_path is required for 'csv' or 'text' output_type.")
    if args.output_type == 'huggingface' and not args.hf_output_repo_id:
        parser.error("--hf_output_repo_id is required for 'huggingface' output_type.")


    # --- Load Data ---
    print(f"Loading data from '{args.input_path}' (type: {args.input_type})...")
    data = None
    if args.input_type == 'huggingface':
        # Load the dataset. This will be a DatasetDict if it has splits.
        data_loaded = load_dataset(args.input_path)
        
        # Determine which split to check for column names. Use 'train' if exists, otherwise first available.
        first_split_name = next(iter(data_loaded.keys()))
        first_split_columns = data_loaded[first_split_name].column_names

        text_column_found = None
        if args.hf_text_column:
            # User provided a specific column name
            if args.hf_text_column in first_split_columns:
                text_column_found = args.hf_text_column
            else:
                raise ValueError(
                    f"Specified Hugging Face text column '{args.hf_text_column}' not found in dataset "
                    f"'{args.input_path}'. Available columns in '{first_split_name}' split: {first_split_columns}"
                )
        else:
            # Fallback to inference if no specific column name was provided
            if 'sentence_ref' in first_split_columns:
                text_column_found = 'sentence_ref'
            else:
                for col_name in first_split_columns:
                    if 'text' in col_name.lower() or 'sentence' in col_name.lower():
                        text_column_found = col_name
                        break
                if text_column_found is None:
                    raise ValueError(
                        f"Could not find a suitable text column in Hugging Face dataset "
                        f"'{args.input_path}'. Please ensure it has 'sentence_ref' or "
                        f"use --hf_text_column to specify it manually. Available columns in '{first_split_name}' split: {first_split_columns}"
                    )

        processed_datasets = DatasetDict()
        for split_name, ds_split in data_loaded.items():
            print(f"Processing split: {split_name}...")
            # Rename the text column to a generic name for the mapping function
            ds_split = ds_split.rename_column(text_column_found, 'text_to_phonetise')
            
            # Add an 'ID' column if not present. Important for CSV output if not HF output.
            if 'ID' not in ds_split.column_names:
                ds_split = ds_split.add_column('ID', [f"{split_name}_{i}" for i in range(len(ds_split))])
            
            processed_datasets[split_name] = ds_split.map(get_phoneme, batched=True, remove_columns=['text_to_phonetise'])
        data = processed_datasets # Store the processed DatasetDict

    elif args.input_type == 'csv':
        df = pd.read_csv(args.input_path)
        # Ensure ID and text columns exist
        if args.column_id not in df.columns:
            raise ValueError(f"CSV input missing expected ID column: '{args.column_id}'")
        if args.column_text not in df.columns:
            raise ValueError(f"CSV input missing expected text column: '{args.column_text}'")

        # Create a Hugging Face Dataset from the DataFrame
        data = Dataset.from_pandas(df[[args.column_id, args.column_text]])
        # Rename columns to generic names for the mapping function
        if 'ID' not in df.columns: 
            data = data.rename_column(args.column_id, 'ID')
        if 'text_to_phonetise' not in df.columns:
            data = data.rename_column(args.column_text, 'text_to_phonetise')
        data = data.map(get_phoneme, batched=True, remove_columns=['text_to_phonetise'])

    elif args.input_type == 'text':
        with open(args.input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()] # Read non-empty lines
        
        # Create a DataFrame for conversion to Dataset
        df = pd.DataFrame({
            'ID': [f"line_{i+1}" for i, _ in enumerate(lines)],
            'text_to_phonetise': lines
        })
        data = Dataset.from_pandas(df)
        data = data.map(get_phoneme, batched=True, remove_columns=['text_to_phonetise'])

    print("Phonetisation complete. Saving results...")

    # --- Save Data ---
    if args.output_type == 'csv':
        if isinstance(data, DatasetDict): # If input was Hugging Face dataset with multiple splits
            # Concatenate all splits into a single pandas DataFrame
            combined_df = pd.concat([ds_split.to_pandas() for ds_split in data.values()])
            combined_df = combined_df[['ID', 'phoneme_sequence']]
            combined_df.to_csv(args.output_path, index=False)
            print(f"Combined results from all splits saved to '{args.output_path}'")
        else: # Single dataset (from CSV or text input, or a single-split HF dataset)
            output_df = data.to_pandas()[['ID', 'phoneme_sequence']]
            output_df.to_csv(args.output_path, index=False)
            print(f"Results saved to '{args.output_path}'")

    elif args.output_type == 'text':
        with open(args.output_path, 'w', encoding='utf-8') as f:
            if isinstance(data, DatasetDict):
                for split_name, ds_split in data.items():
                    for entry in ds_split:
                        if entry['phoneme_sequence'] is not None:
                            f.write(entry['phoneme_sequence'] + '\n')
            else: # Single dataset
                for entry in data:
                    if entry['phoneme_sequence'] is not None:
                        f.write(entry['phoneme_sequence'] + '\n')
        print(f"Results saved to '{args.output_path}' (one phoneme sequence per line).")

    elif args.output_type == 'huggingface':
        if isinstance(data, DatasetDict):
            # Push each split individually
            for split_name, ds_split in data.items():
                print(f"Pushing split '{split_name}' to Hugging Face Hub: '{args.hf_output_repo_id}/{split_name}'...")
                ds_split.push_to_hub(f"{args.hf_output_repo_id}", split=split_name)
        else: # Single dataset (from CSV or text input, or a single-split HF dataset)
            print(f"Pushing dataset to Hugging Face Hub: '{args.hf_output_repo_id}'...")
            data.push_to_hub(args.hf_output_repo_id)
        print("Dataset successfully pushed to Hugging Face Hub.")

    print("Script finished successfully!")

if __name__ == "__main__":
    main()
