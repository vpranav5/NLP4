# This list of imports is likely incomplete --- add anything you need.
# TODO: Your code here.
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.nn.util import sort_batch_by_length
from allennlp.nn.util import replace_masked_values, masked_log_softmax

# Name: Pranav Varanasi
# UTEID: ptv247

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

        # expected EM is 20%
        # rn hidden size is 768, try like powers of 2, dividing by 2, multiplying by 2,
        # and modify the epochs to get above ~20% EM score, check pdf at end, atleast 18% rm

        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(RNN, self).__init__()

        # Initiallize embedding matrix
        self.embedding_matrix = embedding_matrix
        self.num_embedding_words = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)

        # only change hidden size for gru, affine transformation should be for full hidden size
        # dividing by 2 using integer division, in python thats //
        half_hidden = hidden_size // 2

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
        # Use bidirectional GRU with boolean flag
        self.gruPassage = nn.GRU(self.embedding_dim, half_hidden, batch_first = True, bidirectional = True, dropout = dropout)

        # Make a GRU to encode the question. Note that batch_first=True.
        # TODO: Your code here.
        self.gruQuestion = nn.GRU(self.embedding_dim, half_hidden, batch_first = True, bidirectional = True, dropout = dropout)

        # Affine transform for predicting start index.
        # TODO: Your code here.
        # Change shape here based on bidrectional gru and original hidden size, 3 * hidden_size
        self.start_output_projection = nn.Linear(3 * hidden_size, 1)
    
        # Affine transform for predicting end index.
        # TODO: Your code here.

        # initialize end projection
        self.end_output_projection = nn.Linear(3 * hidden_size, 1)

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

        # keep as float tensor for use in later methods 
        passageLengths = passage_mask.sum(dim=1)

        # Make a LongTensor with the length (number non-padding words
        # in) each question.
        # Shape: ?
        # TODO: Your code here.

        # keep as float tensor by summing along mask dimension for non-padding words
        questionLengths = question_mask.sum(dim=1)

        # Part 1: Embed the passages and the questions.
        # 1.1. Embed the passage.
        # TODO: Your code here.
        # Shape: ?

        # Get stored passage embedding
        embedded_passage = self.embedding(passage)

        # 1.2. Embed the question.
        # TODO: Your code here.
        # Shape: ?

        # Get stored question embedding
        embedded_question = self.embedding(question)

        # Part 2. Encode the embedded passages with the RNN.
        # 2.1. Sort embedded passages by decreasing order of passage_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        # TODO: Your code here.

        # method gives a tuple of outputs
        # (sorted passage, sorted passage lengths, restoration index)
        sorted_passage, sorted_passage_lengths, passage_restoration, _ = sort_batch_by_length(embedded_passage, passageLengths)


        # 2.2. Pack the passages with torch.nn.utils.rnn.pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.

        # packing optimizes out the padding, removes out padding words from passages
        # packed_passage is a pytorch object which nests sequences, converts to 2-d matrix
        packed_passage = pack_padded_sequence(sorted_passage, sorted_passage_lengths, batch_first = True)

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

        # returns tuple again, variable, variable expands tuple 0, 1
        # extract unpadded passages from encoding
        passage_unpacked, lens_unpacked = pad_packed_sequence(passageEncoding, batch_first=True)

        # 2.5. Unsort the unpacked, encoded passage to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.

        # Parameters: (input, dim to index along, original ordering)
        # use restoration indices to get original ordering for unpacked passages
        unsorted_passage = passage_unpacked.index_select(0, passage_restoration)

        # Part 3. Encode the embedded questions with the RNN.
        # 3.1. Sort the embedded questions by decreasing order
        #      of question_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        # TODO: Your code here.

        # Returns tuple of 4 arguments
        sorted_question, sorted_question_lengths, question_restoration, _ = sort_batch_by_length(embedded_question, questionLengths)

        # 3.2. Pack the questions with pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.

        # Pack questions based on padding
        packed_question = pack_padded_sequence(sorted_question, sorted_question_lengths, batch_first = True)

        # 3.3. Encode the questions with the RNN.
        # TODO: Your code here.

        # Encode with question bidirectional GRU
        # output is a packed sequence
        questionEncoding, questionHidden = self.gruQuestion(packed_question)

        # 3.4. Unpack (pad) the questions with pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.

        # extract unpadded questions
        question_unpacked, lens_unpacked = pad_packed_sequence(questionEncoding, batch_first = True)

        # 3.5. Unsort the unpacked, encoded question to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.

        # Unsort using question restoration original ordering
        unsorted_question = question_unpacked.index_select(0, question_restoration)

        # 3.6. Take the average of the GRU hidden states.
        # Hint: Be careful how you treat padding.
        # Shape: ?
        # TODO: Your code here.

        # set padding to 0 in question, question hidden gru can have hidden state for padding index that is not all 0
        # element-wise product mask * unpacked, unsorted question of question,unsqueeze and add dimension to mask so it fits
        questionProduct = question_mask.unsqueeze(-1) * unsorted_question

        # sum up non-padded elements of product and get average of gru states
        questionRepresent = (torch.sum(questionProduct, dim = 1) / questionLengths.unsqueeze(1))

        # Part 4: Combine the passage and question representations by
        # concatenating the passage and question representations with
        # their product.

        # 4.1. Reshape the question encoding to make it
        # amenable to concatenation
        # Shape: (batchsize, max passage length, hidden size)
        # TODO: Your code here.

        # questionEncoding (batchsize, max passage length, hidden size)
        # expand depending on this size
        tiled_encoded_q = questionRepresent.unsqueeze(dim=1).expand_as(
            unsorted_passage)

        # 4.2. Concatenate to make the combined representation.
        # Hint: Use torch.cat
        # Shape:  (batch_size, max_passage_size, 6 * embedding_dim)
        # TODO: Your code here.

        # concatenate the expanded passage and expanded questions as well as product over last dim
        combined_x_q = torch.cat([unsorted_passage, tiled_encoded_q,
                                  unsorted_passage * tiled_encoded_q], dim=-1)

        # Part 5: Compute logits for answer start index.

        # 5.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.

        # get start logits with output project, and reshape last column
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

     
