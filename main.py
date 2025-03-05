import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import argparse

def load_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        conll_text = file.read()
        sentences = [sentence.split('\n') for sentence in conll_text.strip().split('\n\n')]
        if file_path.startswith('sample_reference'):  # remove NER tag for sample_reference files to test
            new_sentences = []
            for sentence in sentences:
                new_sentence = []
                for token in sentence:
                    new_sentence.append(token.split(None, 1)[0])
                new_sentences.append(new_sentence)
            sentences = new_sentences

        return sentences


def align_subwords_to_words(labels, word_ids):
    aligned_labels = []
    current_word = None
    current_label = None

    for label, word_id in zip(labels, word_ids):
        if word_id is None:  # Special tokens like [CLS] and [SEP]
            continue

        if word_id != current_word:
            # New word
            if current_word is not None:
                aligned_labels.append(current_label)
            current_word = word_id
            current_label = label

    # Add the last word
    if current_word is not None:
        aligned_labels.append(current_label)

    return aligned_labels


def process_conll_files(input_dir, output_dir, model_name):
    # Load the XLM-RoBERTa model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each CoNLL file in the input directory
    for filename in os.listdir(input_dir):
        if not filename.endswith('.conll2002'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        sentences = load_from_file(input_path)
        text_labels = []
        for sentence in sentences:
            inputs = tokenizer(
                sentence, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                outputs = model(**inputs)

            # Get the predicted labels
            predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
            labels = [model.config.id2label[pred] for pred in predictions]
            word_ids = inputs.word_ids()[1:-1]  # Exclude [CLS] and [SEP] tokens
            labels = align_subwords_to_words(labels[1:-1], word_ids)
            text_labels.append(labels)

        # Write the updated CoNLL data to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence, labels in zip(sentences, text_labels):
                for token, label in zip(sentence, labels):
                    f.write(f'{token} {label}\n')
                f.write('\n')

    print(f"Processed CoNLL files have been saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process CoNLL files using XLM-RoBERTa token classification model")
    parser.add_argument(
        "input_dir", help="Directory containing input CoNLL files", default="reference_nolabels"
    )
    parser.add_argument(
        "output_dir", help="Directory to save processed CoNLL files", default="output"
    )
    parser.add_argument(
        "--model", default="ivlcic/xlmr-ner-slavic",
        help="XLM-RoBERTa model name (default: ivlcic/xlmr-ner-slavic)"
    )
    args = parser.parse_args()

    process_conll_files(args.input_dir, args.output_dir, args.model)

if __name__ == "__main__":
    main()
