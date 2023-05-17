from torch.utils.data import DataLoader, Dataset
import torch
import random


class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs) -> None:
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 60
        self.max_paragraph_len = 150

        # doc_stride
        self.doc_stride = 50

        # input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_sequence_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question['paragraph_id']]

        # pre-processing
        if self.split == 'train':
            answer_start_token = tokenized_paragraph.char_to_token(question['answer_start'])
            answer_end_token = tokenized_paragraph.char_to_token(question['answer_end'])


            ## to-do random slice
            answer_len = answer_end_token - answer_start_token
            random_cut = random.randint(0,(self.max_paragraph_len - answer_len))

            paragraph_start = max(0, min(answer_start_token - random_cut, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len

            input_ids_question = [101] + tokenized_question.ids[: self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]

            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start

            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                input_ids_question = [101] + tokenized_question.ids[: self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]

                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        padding_len = self.max_sequence_len - len(input_ids_question) - len(input_ids_paragraph)
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        return input_ids, token_type_ids, attention_mask

