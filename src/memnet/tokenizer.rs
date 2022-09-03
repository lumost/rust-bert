use std::collections::HashMap;

use rust_tokenizers::{Token, TokenRef};
use rust_tokenizers::error::TokenizerError;
use rust_tokenizers::tokenizer::{BaseTokenizer, MultiThreadedTokenizer, Tokenizer};
use rust_tokenizers::vocab::{BertVocab, Gpt2Vocab, Vocab};

use crate::bert::BertVocabResources;
use crate::gpt2::Gpt2VocabResources;
use crate::resources::{RemoteResource, ResourceProvider};

pub struct MemnetTokenizer {
    tokenizer: BaseTokenizer<MemnetVocab>,
}

impl MemnetTokenizer {
    pub fn build() -> Result<MemnetTokenizer, TokenizerError> {
        let vocab = MemnetVocab::from_file("unused")?;
        Ok(MemnetTokenizer {
            tokenizer:  BaseTokenizer::from_existing_vocab(vocab, true, false)
        })
    }

}

impl Tokenizer<MemnetVocab> for MemnetTokenizer {
    fn vocab(&self) -> &MemnetVocab {
        &MultiThreadedTokenizer::vocab(&self.tokenizer)
    }

    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token> {
        self.tokenizer.tokenize_to_tokens(text)
    }

}

impl MultiThreadedTokenizer<MemnetVocab> for MemnetTokenizer {}

pub struct MemnetVocab {
    encoder: Box<BertVocab>,
    decoder: Box<Gpt2Vocab>,
}

impl MemnetVocab {
    pub fn eos_value(&self) -> &'static str {
        return Gpt2Vocab::eos_value();
    }

    pub fn pad_value() -> &'static str {
        BertVocab::pad_value()
    }

    pub fn bos_value() -> &'static str {
        Gpt2Vocab::bos_value()
    }

    pub fn sep_value() -> &'static str {
        BertVocab::sep_value()
    }
}

impl Vocab for MemnetVocab {
    fn unknown_value() -> &'static str {
        "[UNK]"
    }

    fn get_unknown_value(&self) -> &'static str {
        self.encoder.get_unknown_value()
    }

    fn values(&self) -> &HashMap<String, i64> {
        self.encoder.values()
    }

    fn indices(&self) -> &HashMap<i64, String> {
        self.decoder.indices()
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        self.encoder.special_values()
    }

    fn special_indices(&self) -> &HashMap<i64, String> {
        self.decoder.special_indices()
    }

    fn from_file(path: &str) -> std::result::Result<Self, TokenizerError>
        where
            Self: Sized,
    {
        let encoder_resource = RemoteResource::from_pretrained(BertVocabResources::BERT)
            .get_local_path()
            .unwrap();
        let decoder_resource = RemoteResource::from_pretrained(Gpt2VocabResources::GPT2)
            .get_local_path()
            .unwrap();
        let encoder_vocab = Box::new(BertVocab::from_file(encoder_resource.to_str().unwrap())?);
        let decoder_vocab = Box::new(Gpt2Vocab::from_file(decoder_resource.to_str().unwrap())?);
        Ok(MemnetVocab {
            encoder: encoder_vocab,
            decoder: decoder_vocab,
        })
    }

    fn token_to_id(&self, token: &str) -> i64 {
        self.encoder.token_to_id(token)
    }

    fn id_to_token(&self, id: &i64) -> String {
        self.decoder.id_to_token(id)
    }
}