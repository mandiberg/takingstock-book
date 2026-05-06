import sys
import pandas as pd
import nltk

# Download required NLTK data files (only runs if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv1> [<input_csv2> ...]")
        sys.exit(1)

    input_csvs = sys.argv[1:]
    dfs = []

    for input_csv in input_csvs:
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            print(f"Error reading '{input_csv}': {e}")
            sys.exit(1)
        if 'word' not in df.columns:
            print(f"Error: The CSV file '{input_csv}' must have a column named 'word'.")
            sys.exit(1)
        dfs.append(df)
    
    # Merge all dataframes; reset index to avoid confusion
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates before running POS tagging
    merged_df = merged_df.drop_duplicates(subset=['word']).reset_index(drop=True)

    # Map Penn Treebank POS to generalized categories
    def generalize_pos_tag(word):
        word = str(word)
        # If there are multiple words, use only the last word
        tokens = nltk.word_tokenize(word)
        if not tokens:
            return None
        last_word = tokens[-1]
        pos_tag = nltk.pos_tag([last_word])[0][1]

        # Nouns
        if pos_tag.startswith('NN'):
            return 'NN'
        # Verbs
        elif pos_tag.startswith('VB'):
            return 'VB'
        # Adjectives
        elif pos_tag.startswith('JJ'):
            return 'JJ'
        else:
            return pos_tag

    merged_df['POS'] = merged_df['word'].apply(generalize_pos_tag)
    # INSERT_YOUR_CODE
    merged_df = merged_df.sort_values(by="POS").reset_index(drop=True)

    # Output to new CSV (optional: or print to stdout)
    output_csv = "word_with_pos.csv"
    merged_df.to_csv(output_csv, index=False)
    print(f"POS-tagged dataframe saved to {output_csv}")

if __name__ == "__main__":
    main()
