// Copyright 2020 The Facebook AI Research Team Authors
// Copyright 2020-present, the HuggingFace Inc. team.
// Copyright 2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::borrow::Borrow;
use tch::kind::Kind::Int64;
use tch::nn::{embedding, EmbeddingConfig};
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct LearnedPositionalEmbedding {
    embedding: nn::Embedding,
    padding_index: i64,
    offset: i64,
}

impl LearnedPositionalEmbedding {
    pub fn new<'p, P>(
        p: P,
        num_embeddings: i64,
        embedding_dim: i64,
        padding_index: i64,
    ) -> LearnedPositionalEmbedding
    where
        P: Borrow<nn::Path<'p>>,
    {
        let offset = 2;

        let embedding_config = EmbeddingConfig {
            ..Default::default()
        };
        let num_embeddings = num_embeddings + offset;

        let embedding: nn::Embedding =
            embedding(p.borrow(), num_embeddings, embedding_dim, embedding_config);
        LearnedPositionalEmbedding {
            embedding,
            padding_index,
            offset,
        }
    }

    pub fn forward(&self, input: &Tensor, past_key_values_length: i64) -> Tensor {
        let input_shape = input.size();
        let (_, sequence_length) = (input_shape[0], input_shape[1]);
        let positions = Tensor::arange1(
            past_key_values_length,
            past_key_values_length + sequence_length,
            (Int64, input.device()),
        ) + self.offset;
        positions.apply(&self.embedding)
    }
}
