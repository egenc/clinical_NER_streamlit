import json
import os

with open('spark_nlp_for_healthcare.json') as f:
    license_keys = json.load(f)

for k,v in license_keys.items(): 
    # %set_env $k=$v
    os.environ[k] = v

def start(secret):
    builder = SparkSession.builder \
        .appName("Spark NLP Licensed") \
        .master("local[*]") \
        .config("spark.driver.memory", "16G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:"+version) \
        .config("spark.jars", "https://pypi.johnsnowlabs.com/"+secret+"/spark-nlp-jsl-"+jsl_version+".jar")
      
    return builder.getOrCreate()


from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession

from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import sparknlp_jsl
import sparknlp

import streamlit as st

params = {"spark.driver.memory":"16G",
"spark.kryoserializer.buffer.max":"2000M",
"spark.driver.maxResultSize":"2000M"}

spark = sparknlp_jsl.start(license_keys['SECRET'],params=params)

print ("Spark NLP Version :", sparknlp.version())
print ("Spark NLP_JSL Version :", sparknlp_jsl.version())
# Annotator that transforms a text column from dataframe into an Annotation ready for NLP

medical_models = ['ner_clinical_large', "ner_jsl_greedy", 'ner_jsl']


@st.cache(allow_output_mutation=True)
def load_pipeline(MODEL_NAME):
    documentAssembler = DocumentAssembler()\
            .setInputCol("text")\
            .setOutputCol("document")

    # Sentence Detector annotator, processes various sentences per line

    #sentenceDetector = SentenceDetector()\
            #.setInputCols(["document"])\
            #.setOutputCol("sentence")

    sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
            .setInputCols(["document"])\
            .setOutputCol("sentence")
    
    # Tokenizer splits words in a relevant format for NLP
    tokenizer = Tokenizer()\
            .setInputCols(["sentence"])\
            .setOutputCol("token")
    spellModel = ContextSpellCheckerModel\
            .pretrained('spellcheck_clinical', 'en', 'clinical/models')\
            .setInputCols("token")\
            .setOutputCol("checked")

    # Clinical word embeddings trained on PubMED dataset
    word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
            .setInputCols(["sentence","token"])\
            .setOutputCol("embeddings")

    # NER model trained on i2b2 (sampled from MIMIC) dataset
    clinical_ner = MedicalNerModel.pretrained(MODEL_NAME,"en","clinical/models")\
            .setInputCols(["sentence","token","embeddings"])\
            .setOutputCol("ner")

    ner_converter = NerConverter()\
            .setInputCols(["sentence","token","ner"])\
            .setOutputCol("ner_chunk")

    clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
            .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
            .setOutputCol("assertion")

    sbert_embedder = BertSentenceEmbeddings\
            .pretrained("sbiobert_base_cased_mli","en","clinical/models")\
            .setInputCols(["ner_chunk_doc"])\
            .setOutputCol("sbert_embeddings")
 
    cpt_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt","en", "clinical/models") \
            .setInputCols(["ner_chunk", "sbert_embeddings"]) \
            .setOutputCol("resolution")\
            .setDistanceFunction("EUCLIDEAN")

    nlpPipeline = Pipeline(stages=[
            documentAssembler,
            sentenceDetector,
            tokenizer,
            spellModel,
            word_embeddings,
            clinical_ner,
            ner_converter,
            clinical_assertion,
            sbert_embedder,
            cpt_resolver])

    empty_data = spark.createDataFrame([[""]]).toDF("text")

    model = nlpPipeline.fit(empty_data)
    return model



st.title('NER model for Clinical Entities')
st.write("Please select an index which will be passed to a NER model:")
sparknlp_model = st.sidebar.selectbox("Pipeline name", medical_models)
model_load_state = st.info(f"Loading pretrained pipeline '{sparknlp_model}'...")

model_load_state.empty()

def main():
  import pandas as pd
  import re

  df = pd.read_csv('mimic_100_pats.csv', low_memory = False)

  texts = []
  for idx in range(len(df)):
      begin = df.TEXT[idx].find('History of Present Illness:')
      if begin == -1:
          begin = df.TEXT[idx].find('-year-old')
      if begin == -1:
          continue
      end = df.TEXT[idx].find('Past Medical History:')
      text = df.TEXT[idx][begin+len('-year-old'):end]
      texts.append(text)

  idx = st.slider("Index of the patient", 1, len(texts),1)
  st.subheader(texts[idx])

  text = texts[idx]

  
  light_model = LightPipeline(load_pipeline(sparknlp_model))

  light_result = light_model.fullAnnotate(text)
  


  chunks = []
  entities = []
  sentence= []
  begin = []
  end = []

  for n in light_result[0]['ner_chunk']:
          
      begin.append(n.begin)
      end.append(n.end)
      chunks.append(n.result)
      entities.append(n.metadata['entity']) 
      sentence.append(n.metadata['sentence'])
      
      

  df = pd.DataFrame({'chunks':chunks, 'begin': begin, 'end':end, 
                    'sentence_id':sentence, 'entities':entities})

  st.dataframe(df)

if __name__ == '__main__':
  main()