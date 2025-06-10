
# Arabic Phonetiser Script

This Python script phonetises Arabic text from various sources and outputs the phoneme sequences. It uses the phonetiser library.

## Features
- **Input Flexibility**: Process Hugging Face datasets, CSV files (with custom columns), or plain text files.
- **Output Flexibility**: Save to CSV, plain text, or push directly to Hugging Face Hub.
- **Configurable**: Specify text column names for Hugging Face datasets and CSVs.
- **Error Handling**: Provides clear messages for missing inputs or columns.

## Prerequisites
- Python 3.8+
- pip

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Iqra-Eval/MSA_phonetiser.git
   cd MSA_phonetiser
   ```
2. Install packages:
   ```
   pip install datasets pandas
   ```
3. Hugging Face Login (if pushing to Hub):
   ```
   huggingface-cli login
   ```

## Usage
Run the script with command-line arguments:
```
python run_phonetiser.py --input_type [huggingface|csv|text] \
                         --input_path <input_data> \
                         --output_type [csv|text|huggingface] \
                         [--output_path <output_file>] \
                         [--column_id <csv_id_col>] \
                         [--column_text <csv_text_col>] \
                         [--hf_text_column <hf_text_col>] \
                         [--hf_output_repo_id <hf_repo_id>]
```

### Argument Descriptions
- `--input_type`: huggingface, csv, or text (Required)
- `--input_path`: Path to input file or Hugging Face dataset name (Required)
- `--output_type`: csv, text, or huggingface (Required)
- `--output_path`: Output file path (for csv/text output)
- `--column_id`: CSV ID column name (default: ID)
- `--column_text`: CSV Arabic text column name (default: Arabic text)
- `--hf_text_column`: Hugging Face dataset's Arabic text column name (infers if not provided)
- `--hf_output_repo_id`: Hugging Face repo ID for output (for huggingface output)

## Examples
1. **Hugging Face to CSV**
   ```
   python run_phonetiser.py --input_type huggingface --input_path "IqraEval/Iqra_train" \
       --hf_text_column "tashkeel_sentence" --output_type csv --output_path "output_phonemes.csv"
   ```

2. **Hugging Face to Hugging Face Hub**
   ```
   python run_phonetiser.py --input_type huggingface --input_path "IqraEval/Iqra_train" \
       --hf_text_column "tashkeel_sentence" --output_type huggingface \
       --hf_output_repo_id "your-huggingface-username/my-phonetised-arabic-dataset"
   ```
   (Ensure you're logged into Hugging Face CLI)

3. **Custom CSV to Text File**
   ```
   # my_texts.csv: record_id,arabic_sentence_data
   python run_phonetiser.py --input_type csv --input_path "sample.csv" \
       --column_id "record_id" --column_text "arabic_sentence_data" \
       --output_type text --output_path "phoneme_output.txt"
   ```

4. **Plain Text File to CSV**
   ```
   # input_sentences.txt: one sentence per line
   python run_phonetiser.py --input_type text --input_path "input_sentences.txt" \
       --output_type csv --output_path "phonetised_from_text.csv"
   ```

## Important Notes
- Relies on the phonetiser.phonetise_Arabic module.
- Hugging Face authentication is required for Hub pushes.
- Verify all column names carefully.
