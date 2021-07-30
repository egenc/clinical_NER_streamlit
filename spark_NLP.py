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
        

        # .setOutputCol("document")
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


        # Clinical word embeddings trained on PubMED dataset
        word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
                .setInputCols(["sentence","token"])\
                .setOutputCol("embeddings")

        pos_tagger = PerceptronModel()\
                .pretrained("pos_clinical", "en", "clinical/models") \
                .setInputCols(["sentence", "token"])\
                .setOutputCol("pos_tags")
    
        dependency_parser = DependencyParserModel()\
                .pretrained("dependency_conllu", "en")\
                .setInputCols(["sentence", "pos_tags", "token"])\
                .setOutputCol("dependencies")

        clinical_ner_tagger = MedicalNerModel()\
                .pretrained('jsl_ner_wip_greedy_clinical','en','clinical/models')\
                .setInputCols("sentences", "token", "embeddings")\
                .setOutputCol("ner_tags")    

        ner_chunker = NerConverter()\
                .setInputCols(["sentence", "token", "ner_tags"])\
                .setOutputCol("ner_chunks")

        # NER model trained on i2b2 (sampled from MIMIC) dataset
        clinical_ner = MedicalNerModel.pretrained(MODEL_NAME,"en","clinical/models")\
                .setInputCols(["sentence","token","embeddings"])\
                .setOutputCol("ner")

        #####
        ner_converter = NerConverter()\
                .setInputCols(["sentence","token","ner"])\
                .setOutputCol("ner_chunk")
                # .setWhiteList(['PROBLEM'])
        #####
        # c2doc = Chunk2Doc()\
        #         .setInputCols('ner_chunk')\
        #         .setOutputCol("ner_chunk_doc")

        clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
                .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
                .setOutputCol("assertion")

        sbert_embedder = BertSentenceEmbeddings\
                .pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
                .setInputCols(["document"])\
                .setOutputCol("sbert_embeddings")
        
        cpt_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt_procedures_augmented","en", "clinical/models") \
                .setInputCols(["document", "sbert_embeddings"]) \
                .setOutputCol("cpt_code")\
                .setDistanceFunction("EUCLIDEAN")

        re_model = RelationExtractionModel()\
                .pretrained("re_bodypart_directions", "en", 'clinical/models')\
                .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
                .setOutputCol("relations")\
                .setRelationPairs(['direction-external_body_part_or_region', 
                                'external_body_part_or_region-direction',
                                'direction-internal_organ_or_component',
                                'internal_organ_or_component-direction'
                                ])\
                .setMaxSyntacticDistance(4)\
                .setPredictionThreshold(0.9)


        nlpPipeline = Pipeline(stages=[
                documentAssembler,
                sentenceDetector,
                tokenizer,
                word_embeddings,
                pos_tagger,
                dependency_parser,
                clinical_ner_tagger,
                ner_chunker,
                clinical_ner,
                ner_converter,
                clinical_assertion,
                sbert_embedder,
                cpt_resolver,
                re_model])
        
        
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

        idx = st.slider("Index of the patient", 1, 100,1)
        text = df.TEXT[idx]

        # st.subheader(text)

        light_model = LightPipeline(load_pipeline(sparknlp_model))
        light_result = light_model.fullAnnotate(text)

        chunks = []
        entities = []
        status = []
        codes = []
        all_codes = []
        resolutions = []

        rel_pairs=[]

        print('-'*100)
        print(light_result[0]['relations'])
        print('-'*100)
        #   , code , light_result[0]['cpt_code']
        for chunk, assertion, code in zip(light_result[0]['ner_chunk'] ,light_result[0]['assertion'], light_result[0]['cpt_code']):
                
                chunks.append(chunk.result)
                entities.append(chunk.metadata['entity']) 
                status.append(assertion.result)
                codes.append(code.result) 
                all_codes.append(code.metadata['all_k_results'].split(':::'))
                resolutions.append(code.metadata['all_k_resolutions'].split(':::'))

        def get_relations_df (results, col='relations'):
                rel_pairs=[]
                for rel in results[0][col]:
                        rel_pairs.append((
                        rel.result, 
                        rel.metadata['entity1'], 
                        rel.metadata['entity1_begin'],
                        rel.metadata['entity1_end'],
                        rel.metadata['chunk1'], 
                        rel.metadata['entity2'],
                        rel.metadata['entity2_begin'],
                        rel.metadata['entity2_end'],
                        rel.metadata['chunk2'], 
                        rel.metadata['confidence']
                        ))

                rel_df = pd.DataFrame(rel_pairs, columns=['relations',
                                                        'entity1','entity1_begin','entity1_end','chunk1',
                                                        'entity2','entity2_end','entity2_end','chunk2', 
                                                        'confidence'])
                # limit df columns to get entity and chunks with results only
                rel_df = rel_df.iloc[:,[0,1,4,5,8,9]]
                
                return rel_df


        rel_df = get_relations_df(light_result)

        print(len(chunks))
        print(len(entities))
        print(len(status))
        # print(len(codes))
        # print(len(resolutions))

        # df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status})
        df = pd.DataFrame({'chunks':chunks, 'entities':entities, 'assertion':status, 'codes':codes, 'all_codes':all_codes, 'resolutions':resolutions})

        st.dataframe(df)
        st.dataframe(rel_df)


if __name__ == '__main__':
    main()