# This list of imports is likely incomplete --- add anything you need.
# TODO: Your code here.
import torch
import torch.nn as nn
from allennlp.nn.util import masked_softmax

# Name: Pranav Varanasi
# UT EID: ptv247

class AttentionRNN(nn.Module):
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
        super(AttentionRNN, self).__init__()

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
        # Initialize embedding weights
        self.embedding.weight = nn.Parameter(self.embedding_matrix,
                                             requires_grad=False)

        # Make a RNN to encode the passage. Note that batch_first=True.
        # TODO: Your code here.
        # Use GRU variant of RNN
        self.gruPassage = nn.GRU(self.embedding_dim, half_hidden, batch_first = True, bidirectional = True, dropout = dropout)

        # Make a RNN to encode the question. Note that batch_first=True.
        # TODO: Your code here.
        # Create GRU for question
        self.gruQuestion = nn.GRU(self.embedding_dim, half_hidden, batch_first = True, bidirectional = True, dropout = dropout)

        # Affine transform for attention.
        # TODO: Your code here.
        self.attention_transform = nn.Linear(3 * hidden_size, 1)

        # Affine transform for predicting start index.
        # TODO: Your code here.
        # Apply affine transform with 3 * original hidden size
        self.start_output_projection = nn.Linear(3 * hidden_size, 1)

        # Affine transform for predicting end index.
        # TODO: Your code here.
        # Apply end affine transform with 3 * original hidden size
        self.end_output_projection = nn.Linear(3 * hidden_size, 1)

        # Dropout layer
        # TODO: Your code here.

        # Stores the number of gradient updates performed.
        self.global_step = 0

    def forward(self, passage, question):
        """
        The forward pass of the RNN-based model with attention.

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

        # Sum along dim 1 to get length of non-padding words
        passageLengths = passage_mask.sum(dim=1)

        # Make a LongTensor with the length (number non-padding words
        # in) each question.
        # Shape: ?
        # TODO: Your code here.
        
        # Sum along dim 1 to get length of non-padding words for questions
        questionLengths = question_mask.sum(dim=1)

        # Part 1: Embed the passages and the questions.
        # 1.1 Embed the passage.
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

        # Sort training batch with allennlp function with passage and lengths as parameters
        sorted_passage, sorted_passage_lengths, passage_restoration, _ = sort_batch_by_length(embedded_passage, passageLengths)

        # 2.2. Pack the passages with torch.nn.utils.rnn.pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.

        # pack sequence based on non-padding words
        packed_passage = pack_padded_sequence(sorted_passage, sorted_passage_lengths, batch_first = True)

        # 2.3. Encode the packed passages with the RNN.
        # TODO: Your code here.

        # get passage representation from GRU
        passageEncoding, passageHidden = self.gruPassage(packed_passage)

        # 2.4. Unpack (pad) the passages with
        # torch.nn.utils.rnn.pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.

        # Unpack sequence based on padding words
        passage_unpacked, lens_unpacked = pad_packed_sequence(passageEncoding, batch_first=True)

        # 2.5. Unsort the unpacked, encoded passage to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.

        # get unsorted passage using restoration indices
        unsorted_passage = passage_unpacked.index_select(0, passage_restoration)

        # Part 3. Encode the embedded questions with the RNN.
        # 3.1. Sort the embedded questions by decreasing order
        #      of question_lengths.
        # Hint: allennlp.nn.util.sort_batch_by_length might be helpful.
        # TODO: Your code here.

        # Sort question batches
        sorted_question, sorted_question_lengths, question_restoration, _ = sort_batch_by_length(embedded_question, questionLengths)

        # 3.2. Pack the questions with pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.

        # pack questions based on padding words
        packed_question = pack_padded_sequence(sorted_question, sorted_question_lengths, batch_first = True)

        # 3.3. Encode the questions with the RNN.
        # TODO: Your code here.

        # Get question representation from GRU
        questionEncoding, questionHidden = self.gruQuestion(packed_question)

        # 3.4. Unpack (pad) the questions with pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.

        # Unpack question representation based on padding words
        question_unpacked, lens_unpacked = pad_packed_sequence(questionEncoding, batch_first = True)

        # 3.5. Unsort the unpacked, encoded question to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.

        # restore original question ordering using restoration indices
        unsorted_question = question_unpacked.index_select(0, question_restoration)


        # Part 4. Calculate attention weights and attend to question.

        # use softmax to convert weights to probabilities, use those probabilites u multiply questions by to get weighted average
        # you will have another affine transformation to get the weights from the passages, trainable parameter
        # define another matrix for attention and multiply that by tensor to get logits, then do softmax from logits to get probability
        # then computed weighted average from that probability

        # 4.1. Expand the encoded question to shape suitable for attention.
        # Hint: Think carefully about what the shape of the attention
        # input vector should be. torch.unsqueeze and torch.expand
        # might be useful.
        # Shape: (batch_size, passage_len, question_len, hidden_dim) (64, 15, 124, 512)
        # need to get question and passage shape the same
        # TODO: Your code here.

        # Goal: (batch_size, passage_len, question_len, hidden_dim) (64, 15, 124, 512)

        # unsorted question shape is (batch_size, question_len, hidden_dim) (64, , 15 x 512)
        # passage_mask.size(1) would get u max passage size for the mask
        expanded_question = unsorted_question.unsqueeze(dim=1).expand(-1,passage_mask.size(1), -1, -1)

        # 4.2. Expand the encoded passage to shape suitable for attention.
        # Hint: Think carefully about what the shape of the attention
        # input vector should be. torch.unsqueeze and torch.expand
        # might be useful.
        # Shape: ?
        # TODO: Your code here.

        # unsorted pasage shape is (batch_size, passage_len, hidden_dim) (64, 124, 512)
        expanded_passage = unsorted_passage.unsqueeze(dim=2).expand(-1, -1, question_mask.size(1),-1)


        # 4.3. Build attention_input. This is the tensor passed through
        # the affine transform.
        # Hint: Think carefully what the shape of this tensor should be.
        # torch.cat might be useful.
        # Shape: ?

        # attention_input is the concatenating of 4.1 and 4.2
        # pass in attention input to affine transformatioon
        # use affine_transform function and pass in the concatenated matrix
        # TODO: Your code here.
        # concatenate along the last dimension
        attention_input = torch.cat([expanded_passage, expanded_question,
                                  expanded_passage * expanded_question], dim=-1)

        # 4.4. Apply affine transform to attention input to get
        # attention logits. You will need to slightly reshape it
        # into a tensor of the shape you expect.
        # Shape: 
        # TODO: Your code here.

        # apply affine transforms, reshape the last dimension
        attention_logits = self.attention_tranform(attention_input).squeeze(-1)

        # 4.5. Masked-softmax the attention logits over the last dimension
        # to normalize and make the attention logits a proper
        # probability distribution.
        # Hint: allennlp.nn.util.last_dim_softmax might be helpful.
        # Shape: ?
        # TODO: Your code here.

        prob_dist = masked_softmax(attention_logits, question_mask, dim = -1)

        # 4.6. Use the attention weights to get a weighted average
        # of the RNN output from encoding the question for each
        # passage word.
        # Hint: torch.bmm might be helpful.
        # Shape: ?
        # use torch.bmm with question to add weights
        # TODO: Your code here.

        # compute weighted average using matrix product
        attentionWeights = torch.bmm(prob_dist, unsorted_question)

        # Part 5: Combine the passage and question representations by
        # concatenating the passage and question representations with
        # their product.
        # 5.1. Concatenate to make the combined representation.
        # Hint: Use torch.cat
        # Shape: ?
        # TODO: Your code here.

        combinedRepresent = torch.cat([unsorted_passage, attentionWeights,
                                unsorted_passage * attentionWeights], dim=-1)

        # Part 6: Compute logits for answer start index.

        # 6.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.

        start_logits = self.start_output_projection(combinedRepresent).squeeze(-1)

        # 6.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your start_logits.
        # Hint: allennlp.nn.util.replace_masked_values might be helpful.
        # Shape: ?
        # TODO: Your code here.

        start_logits = replace_masked_values(start_logits, passage_mask, -1e7)


        # 6.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_start_logits.
        # Hint: allennlp.nn.util.masked_log_softmax might be helpful.
        # Shape: ?
        # TODO: Your code here.
        
        softmax_start_logits = masked_log_softmax(start_logits, passage_mask)

        # Part 7: Compute logits for answer end index.

        # 7.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.

        end_logits = self.end_output_projection(combinedRepresent).squeeze(-1)

        # 7.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your end_logits.
        # Hint: allennlp.nn.util.replace_masked_values might be helpful.
        # Shape: ?
        # TODO: Your code here.

        end_logits = replace_masked_values(end_logits, passage_mask, -1e7)

        # 7.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_end_logits.
        # Hint: allennlp.nn.util.masked_log_softmax might be helpful.
        # Shape: ?
        # TODO: Your code here.

        softmax_end_logits = masked_log_softmax(end_logits, passage_mask)

        # Part 8: Output a dictionary with the start_logits, end_logits,
        # softmax_start_logits, softmax_end_logits.
        # TODO: Your code here. Remove the NotImplementedError below.
        
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "softmax_start_logits": softmax_start_logits,
            "softmax_end_logits": softmax_end_logits
        }
