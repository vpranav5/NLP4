# This list of imports is likely incomplete --- add anything you need.
# TODO: Your code here.
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.nn.util import sort_batch_by_length
from allennlp.nn.util import replace_masked_values, masked_log_softmax

# Name: Pranav Varanasi
# UT EID: ptv247
# Changed RNN File

class RNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size,
                 dropout):
        """
        Parameters
        ----------
        embedding_matrix: FloatTensor
            FloatTensor matrix of shape (num_words, embedding_dim),
            where each row of the matrix is a word vector for the
            associated word index.

        hidden_size: int
            The size of the hidden state in the RNN.

        dropout: float
            The dropout rate.
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(RNN, self).__init__()

        self.embedding_matrix = embedding_matrix
        self.num_embedding_words = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)

        hidden_size = hidden_size // 2
        # dividing by 2 using integer division, in python thats //

        # Create Embedding object
        # TODO: Your code here.
        self.embedding = nn.Embedding(self.num_embedding_words,
                                      self.embedding_dim, padding_idx=0)

        # Load our embedding matrix weights into the Embedding object,
        # and make them untrainable (requires_grad=False)
        # TODO: Your code here.
        self.embedding.weight = nn.Parameter(self.embedding_matrix,
                                             requires_grad=False)

        # Make a GRU to encode the passage. Note that batch_first=True.
        # TODO: Your code here.
        self.gruPassage = nn.GRU(self.embedding_dim, hidden_size, batch_first = True, bidirectional = True, dropout = dropout)
        #self.gruPassage = nn.GRU(self.embedding_dim, hidden_size, batch_first = True, dropout = dropout)

        # Make a GRU to encode the question. Note that batch_first=True.
        # TODO: Your code here.
        self.gruQuestion = nn.GRU(self.embedding_dim, hidden_size, batch_first = True, bidirectional = True, dropout = dropout)

        #self.gruQuestion = nn.GRU(self.embedding_dim, hidden_size, batch_first = True, dropout = dropout)

        # Affine transform for predicting start index.
        # TODO: Your code here.
        # Change shape here based on bidrectional gru, 6 * hidden size
        self.start_output_projection = nn.Linear(3 * hidden_size, 1)
        #self.start_output_projection = nn.Linear(3 * self.embedding_dim, 1)

        # Affine transform for predicting end index.
        # TODO: Your code here.
        self.end_output_projection = nn.Linear(3 * hidden_size, 1)
        #self.end_output_projection = nn.Linear(3 * self.embedding_dim, 1)

        # Dropout layer
        # TODO: Your code here.
        #self.dropout = nn.Dropout(p = dropout)

        # Stores the number of gradient updates performed.
        self.global_step = 0

    def forward(self, passage, question):
        """
        The forward pass of the RNN-based model.

        Parameters
        ----------
        passage: Variable(LongTensor)
            A Variable(LongTensor) of shape (batch_size, passage_length)
            representing the words in the passage for each batch.

        question: Variable(LongTensor)
            A Variable(LongTensor) of shape (batch_size, question_length)
            representing the words in the question for each batch.

        Returns
        -------
        An output dictionary consisting of:
        start_logits: Variable(FloatTensor)
            The first element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Each value is the score
            assigned to a given token. Masked indices are assigned very
            small scores (-1e7).

        end_logits: Variable(FloatTensor)
            The second element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Each value is the score
            assigned to a given token. Masked indices are assigned very
            small scores (-1e7).

        softmax_start_logits: Variable(FloatTensor)
            The third element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Exactly the same as
            start_logits, but with a masked log softmax applied. Represents
            a probability distribution over the passage, indicating the
            probability that any given token is where the answer begins.
            Masked indices have probability mass of -inf.

        softmax_end_logits: Variable(FloatTensor)
            The fourth element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Exactly the same as
            start_logits, but with a masked log softmax applied. Represents
            a probability distribution over the passage, indicating the
            probability that any given token is where the answer end.
            Masked indices have probability mass of -inf.
        """
        # Mask: FloatTensor with 0 in positions that are
        # padding (word index 0) and 1 in positions with actual words.
        # Make a mask for the passage. Shape: ?
        # TODO: Your code here.
        passage_mask = (passage != 0).type(
            torch.cuda.FloatTensor if passage.is_cuda else
            torch.FloatTensor)

        # Make a mask for the question. Shape: ?
        # TODO: Your code here.
        question_mask = (question != 0).type(
            torch.cuda.FloatTensor if question.is_cuda else
            torch.FloatTensor)

        # Make a LongTensor with the length (number non-padding words
        # in) each passage.
        # Shape: ?
        # TODO: Your code here.
        # keep as float tensor
        passageLengths = passage_mask.sum(dim=1)

        # Make a LongTensor with the length (number non-padding words
        # in) each question.
        # Shape: ?
        # TODO: Your code here.
        questionLengths = question_mask.sum(dim=1)

        # Part 1: Embed the passages and the questions.
        # 1.1. Embed the passage.
        # TODO: Your code here.
        # Shape: ?
        embedded_passage = self.embedding(passage)

        # 1.2. Embed the question.
        # TODO: Your code here.
        # Shape: ?
        embedded_question = self.embedding(question)

        # Part 2. Encode the embedded passages with the RNN.
        # 2.1. Sort embedded passages by decreasing order of passage_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        # TODO: Your code here.
        sorted_passage = sort_batch_by_length(embedded_passage, passageLengths)

        # 2.2. Pack the passages with torch.nn.utils.rnn.pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.
        # packing optimizes out the padding, removes out padding words from passages, look at stack overflow
        # packed_passage is a pytorch object which nests sequences, converts to 2-d matrix
        packed_passage = pack_padded_sequence(sorted_passage, passageLengths, batch_first = True)

        # 2.3. Encode the packed passages with the RNN.
        # TODO: Your code here. (input), feeding in optimized passages thru the network nodes
        # encoding is used to represent input within the neural network
        # output is a packed sequence
        passageEncoding, passageHidden = self.gruPassage(packed_passage)

        # 2.4. Unpack (pad) the passages with
        # torch.nn.utils.rnn.pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.
        passage_unpacked, lens_unpacked = pad_packed_sequence(passageEncoding, batch_first=True)

        # 2.5. Unsort the unpacked, encoded passage to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.
        # Parameters: (input, dim to index along, original ordering)
        unsorted_passage = torch.index_select(passage_unpacked, 0, passageLengths)

        # Part 3. Encode the embedded questions with the RNN.
        # 3.1. Sort the embedded questions by decreasing order
        #      of question_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        # TODO: Your code here.
        sorted_question = sort_batch_by_length(embedded_question, questionLengths)

        # 3.2. Pack the questions with pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.
        packed_question = pack_padded_sequence(sorted_question, questionLengths, batch_first = True)

        # 3.3. Encode the questions with the RNN.
        # TODO: Your code here.
        # Encode with question bidirectional GRU
        # output is a packed sequence
        questionEncoding, questionHidden = self.gruQuestion(packed_question)

        # 3.4. Unpack (pad) the questions with pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.
        question_unpacked, lens_unpacked = pad_packed_sequence(questionEncoding, batch_first = True)

        # 3.5. Unsort the unpacked, encoded question to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.
        # Unsort using questionLengths original ordering
        unsorted_question = torch.index_select(question_unpacked, 0, questionLengths)

        # 3.6. Take the average of the GRU hidden states.
        # Hint: Be careful how you treat padding.
        # Shape: ?
        # TODO: Your code here.
        questionRepresent = (torch.sum(questionHidden, dim = 1) / questionLengths.unsqueeze(1))

        # Part 4: Combine the passage and question representations by
        # concatenating the passage and question representations with
        # their product.

        # 4.1. Reshape the question encoding to make it
        # amenable to concatenation
        # Shape: ?
        # TODO: Your code here.

        tiled_encoded_q = questionRepresent.unsqueeze(dim=1).expand_as(
            embedded_passage)

        # 4.2. Concatenate to make the combined representation.
        # Hint: Use torch.cat
        # Shape: ?
        # TODO: Your code here.

        # Shape: (batch_size, max_passage_size, 6 * embedding_dim)
        combined_x_q = torch.cat([embedded_passage, tiled_encoded_q,
                                  embedded_passage * tiled_encoded_q], dim=-1)

        # Part 5: Compute logits for answer start index.

        # 5.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.

        start_logits = self.start_output_projection(combined_x_q).squeeze(-1)
        

        # 5.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your start_logits.
        # Hint: allennlp.nn.util.replace_masked_values might be helpful.
        # Shape: ?
        # TODO: Your code here.

        start_logits = replace_masked_values(start_logits, passage_mask, -1e7)

        # 5.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_start_logits.
        # Hint: allennlp.nn.util.masked_log_softmax might be helpful.
        # Shape: ?
        # TODO: Your code here.

        softmax_start_logits = masked_log_softmax(start_logits, passage_mask)

        # Part 6: Compute logits for answer end index.

        # 6.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.
        end_logits = self.end_output_projection(combined_x_q).squeeze(-1)

        # 6.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your end_logits.
        # Hint: allennlp.nn.util.replace_masked_values might be helpful.
        # Shape: ?
        # TODO: Your code here.
        end_logits = replace_masked_values(end_logits, passage_mask, -1e7)

        # 6.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_end_logits.
        # Hint: allennlp.nn.util.masked_log_softmax might be helpful.
        # Shape: ?
        # TODO: Your code here.
        softmax_end_logits = masked_log_softmax(end_logits, passage_mask)

        # Part 7: Output a dictionary with the start_logits, end_logits,
        # softmax_start_logits, softmax_end_logits.
        # TODO: Your code here. Remove the NotImplementedError below.

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "softmax_start_logits": softmax_start_logits,
            "softmax_end_logits": softmax_end_logits
        }

     
