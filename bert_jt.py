import jittor
from jittor import nn
import math
# from transformers import BertConfig, BertModel, BertTokenizer
import json

ACT2FN = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "tanh": jittor.tanh,
    "sigmoid": jittor.sigmoid,
}

class BertConfig():
    def __init__(self, 
                 model_type='bert', 
                 vocab_size=30522, 
                 hidden_size=768, 
                 num_hidden_layers=12, 
                 num_attention_heads=12, 
                 intermediate_size=3072, 
                 hidden_act='gelu', 
                 hidden_dropout_prob=0.1, 
                 attention_probs_dropout_prob=0.1, 
                 max_position_embeddings=512, 
                 type_vocab_size=2, 
                 initializer_range=0.02, 
                 layer_norm_eps=1e-12, 
                 pad_token_id=0, 
                 position_embedding_type='absolute', 
                 use_cache=True, 
                 gradient_checkpointing=False, 
                 transformers_version='4.2.1',
                 **kwargs):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        self.transformers_version = transformers_version
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @classmethod
    def from_pretrained_file(cls, file_path):
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        # 根据加载的配置字典初始化 BertConfig 对象
        return cls(**config_dict)


class BertBase(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        # Save config
        self.config = config

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.trunc_normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.trunc_normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.zero_()
            module.weight.constant_(1.0)

            
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = jittor.arange(config.max_position_embeddings).expand((1, 1))

    def execute(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
    ):

        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = jittor.zeros(input_shape)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def execute(
            self,
            hidden_states,
            attention_mask=None,
    ):
        
        all_hidden_states = ()
        for _, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs
        
        all_hidden_states = all_hidden_states + (hidden_states,)    

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states
        }
        
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def execute(
            self,
            hidden_states,
            attention_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )

        intermediate_output = self.intermediate(self_attention_outputs)
        layer_output = self.output(intermediate_output, self_attention_outputs)

        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def execute(
            self,
            hidden_states,
            attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )

        attention_output = self.output(self_outputs, hidden_states)
        outputs = attention_output
        return outputs


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def execute(
            self,
            hidden_states,
            attention_mask=None,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = jittor.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel execute() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = jittor.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def execute(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def execute(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def execute(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def execute(self, hidden_states):
        # "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(BertBase):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()

    def execute(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            output_hidden_states=False
    ):
        input_shape = input_ids.size()

        if attention_mask is None:
            attention_mask = jittor.ones(input_shape)

        # Make it broadcastable to all heads. This is for compute attention_scores.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs["last_hidden_state"]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs["hidden_states"] if output_hidden_states else None
        }

    def get_extended_attention_mask(self, attention_mask):
        """
        Makes broadcastable attention
        Arguments:
            attention_mask (`jittor.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        Returns:
            `jittor.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = None
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # Make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
