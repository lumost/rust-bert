// Copyright 2020, Microsoft and the HuggingFace Inc. team.
// Copyright 2022 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::common::dropout::XDropout;
use crate::deberta::{PositionAttentionType, PositionAttentionTypes};
use crate::deberta_v2::DebertaV2Config;
use crate::RustBertError;
use std::borrow::Borrow;
use tch::{nn, Device, Kind, Tensor};

pub fn make_log_bucket_position(
    relative_pos: &Tensor,
    bucket_size: i64,
    max_position: i64,
) -> Tensor {
    let sign = relative_pos.sign();
    let mid = bucket_size / 2;
    let abs_pos = relative_pos.abs().where_scalarother(
        &relative_pos
            .lt(mid)
            .logical_and(&relative_pos.gt(-mid))
            .logical_not(),
        mid - 1,
    );
    let log_pos = (((abs_pos / mid).log() / (((max_position - 1) / mid) as f64).ln()) * (mid - 1))
        .ceil()
        + mid;
    let bucket_pos = relative_pos
        .where_self(&abs_pos.less_equal(mid), &(log_pos * sign))
        .to_kind(Kind::Int64);
    bucket_pos
}

pub fn build_relative_position(
    query_size: i64,
    key_size: i64,
    bucket_size: i64,
    max_position: i64,
    device: Device,
) -> Tensor {
    let q_ids = Tensor::arange(query_size, (Kind::Int64, device));
    let k_ids = Tensor::arange(key_size, (Kind::Int64, device));
    let mut rel_pos_ids = q_ids.unsqueeze(-1) - k_ids.tile(&[q_ids.size()[0], 1]);
    if (bucket_size > 0) & (max_position > 0) {
        rel_pos_ids = make_log_bucket_position(&rel_pos_ids, bucket_size, max_position);
    }
    rel_pos_ids.slice(0, 0, query_size, 1).unsqueeze(0)
}

pub struct DisentangledSelfAttention {
    query_proj: nn::Linear,
    key_proj: nn::Linear,
    value_proj: nn::Linear,
    pos_key_proj: Option<nn::Linear>,
    pos_query_proj: Option<nn::Linear>,
    position_buckets: Option<i64>,
    pos_embed_size: Option<i64>,
    dropout: XDropout,
    num_attention_heads: i64,
    pos_att_type: PositionAttentionTypes,
    max_relative_positions: Option<i64>,
    pos_dropout: Option<XDropout>,
    output_attentions: bool,
    share_attention_key: bool,
}

impl DisentangledSelfAttention {
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DisentangledSelfAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let num_attention_heads = config.num_attention_heads;
        let query_proj = nn::linear(
            p / "query_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let key_proj = nn::linear(
            p / "key_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let value_proj = nn::linear(
            p / "value_proj",
            config.hidden_size,
            config.hidden_size,
            Default::default(),
        );
        let share_attention_key = config.share_att_key.unwrap_or(false);
        let pos_att_type = config.pos_att_type.clone().unwrap_or_default();
        let relative_attention = config.relative_attention.unwrap_or(false);

        let (
            max_relative_positions,
            pos_dropout,
            pos_key_proj,
            pos_query_proj,
            position_buckets,
            pos_embed_size,
        ) = if relative_attention {
            let position_buckets = config.position_buckets.unwrap_or(-1);
            let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings;
            }
            let pos_ebd_size = if position_buckets > 0 {
                position_buckets
            } else {
                max_relative_positions
            };
            let pos_dropout = Some(XDropout::new(config.hidden_dropout_prob));

            let (pos_key_proj, pos_query_proj) = if !share_attention_key {
                let pos_key_proj = if pos_att_type.has_type(PositionAttentionType::c2p)
                    | pos_att_type.has_type(PositionAttentionType::p2p)
                {
                    Some(nn::linear(
                        p / "pos_key_proj",
                        config.hidden_size,
                        config.hidden_size,
                        Default::default(),
                    ))
                } else {
                    None
                };
                let pos_query_proj = if pos_att_type.has_type(PositionAttentionType::p2c)
                    | pos_att_type.has_type(PositionAttentionType::p2p)
                {
                    Some(nn::linear(
                        p / "pos_query_proj",
                        config.hidden_size,
                        config.hidden_size,
                        Default::default(),
                    ))
                } else {
                    None
                };
                (pos_key_proj, pos_query_proj)
            } else {
                (None, None)
            };

            (
                Some(max_relative_positions),
                pos_dropout,
                pos_key_proj,
                pos_query_proj,
                Some(position_buckets),
                Some(pos_ebd_size),
            )
        } else {
            (None, None, None, None, None, None)
        };
        let dropout = XDropout::new(config.attention_probs_dropout_prob);

        let output_attentions = config.output_attentions.unwrap_or(false);

        DisentangledSelfAttention {
            query_proj,
            key_proj,
            num_attention_heads,
            pos_att_type,
            max_relative_positions,
            pos_dropout,
            dropout,
            output_attentions,
            value_proj,
            pos_key_proj,
            pos_query_proj,
            position_buckets,
            pos_embed_size,
            share_attention_key,
        }
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Tensor {
        let mut new_shape = x.size();
        let _ = new_shape.pop();
        new_shape.extend_from_slice(&[self.num_attention_heads, -1]);
        let x = x.view(new_shape.as_slice());
        x.permute(&[0, 2, 1, 3])
            .contiguous()
            .view([-1, x.size()[1], x.size().last().unwrap()])
    }

    fn disentangled_att_bias(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        relative_pos: Option<&Tensor>,
        relative_embeddings: &Tensor,
        scale_factor: f64,
    ) -> Result<Tensor, RustBertError> {
        let mut key_layer_size = key_layer.size();
        key_layer_size.reverse();
        let mut query_layer_size = query_layer.size();
        query_layer_size.reverse();

        let calc_relative_pos = if relative_pos.is_none() {
            Some(build_relative_position(
                query_layer_size[1],
                key_layer_size[1],
                self.position_buckets.unwrap_or(-1),
                self.max_relative_positions.unwrap_or(-1),
                query_layer.device(),
            ))
        } else {
            None
        };
        let relative_pos = relative_pos.unwrap_or_else(|| calc_relative_pos.as_ref().unwrap());
        let relative_pos = match &relative_pos.dim() {
            2 => relative_pos.unsqueeze(0).unsqueeze(0),
            3 => relative_pos.unsqueeze(1),
            4 => relative_pos.shallow_clone(),
            _ => {
                return Err(RustBertError::ValueError(format!(
                    "Expected relative position of dimensions 2, 3 or 4, got {}",
                    relative_pos.dim()
                )))
            }
        };

        // This method only gets called if relative attention is True
        let att_span = self.pos_embed_size.unwrap();
        let relative_embeddings = relative_embeddings
            .slice(0, 0, 2 * att_span, 1)
            .unsqueeze(0);

        let key_proj = self.pos_key_proj.as_ref().unwrap_or(&self.key_proj);
        let query_proj = self.pos_query_proj.as_ref().unwrap_or(&self.query_proj);

        let pos_query_layer = self
            .transpose_for_scores(&relative_embeddings.apply(query_proj))
            .repeat(&[query_layer.size()[0] / self.num_attention_heads, 1, 1]);
        let pos_key_layer = self
            .transpose_for_scores(&relative_embeddings.apply(key_proj))
            .repeat(&[query_layer.size()[0] / self.num_attention_heads, 1, 1]);

        Ok(Tensor::new())
    }
}
