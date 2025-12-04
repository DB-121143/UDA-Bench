-- Query 1: 1 (CSPaper)
SELECT agent_framework, application_domain, performance_on_hotpotqa FROM CSPaper WHERE performance_on_hotpotqa != 'KILT-EM: 27.3 (T5-Base)';

-- Query 2: 1 (CSPaper)
SELECT generator_model, reasoning_depth, data_modality FROM CSPaper WHERE data_modality = 'proprietary enterprise dataset';

-- Query 3: 1 (CSPaper)
SELECT uses_knowledge_graph, uses_reranker, use_agent FROM CSPaper WHERE uses_knowledge_graph <= 'Yes';

-- Query 4: 1 (CSPaper)
SELECT baseline, uses_knowledge_graph, evaluation_dataset FROM CSPaper WHERE evaluation_dataset = 'QAMPARI';

-- Query 5: 1 (CSPaper)
SELECT evaluation_dataset, baseline, use_agent FROM CSPaper WHERE baseline = 'FLARE';

-- Query 6: 1 (CSPaper)
SELECT evaluation_dataset, reasoning_depth, data_modality FROM CSPaper WHERE reasoning_depth = 'single-hop';

-- Query 7: 1 (CSPaper)
SELECT data_modality, multi_turn_retrieval, evaluation_metric FROM CSPaper WHERE evaluation_metric = 'F1-score';

-- Query 8: 1 (CSPaper)
SELECT evaluation_metric, baseline, use_agent FROM CSPaper WHERE use_agent = 'No';

-- Query 9: 1 (CSPaper)
SELECT retrieval_method, agent_framework, baseline FROM CSPaper WHERE baseline = 'Single-step Approach';

-- Query 10: 1 (CSPaper)
SELECT topic, reasoning_depth, performance_on_hotpotqa FROM CSPaper WHERE topic != 'LLM';

-- Query 11: 2 (CSPaper)
SELECT performance_on_NQ, agent_framework, topic FROM CSPaper WHERE performance_on_NQ != 'MH_test: 50.0' AND generator_model != 'GPT-4o-mini';

-- Query 12: 2 (CSPaper)
SELECT evaluation_metric, performance_on_hotpotqa, application_domain FROM CSPaper WHERE application_domain = 'Academic' AND evaluation_metric = 'recall';

-- Query 13: 2 (CSPaper)
SELECT performance_on_hotpotqa, evaluation_metric, retrieval_method FROM CSPaper WHERE performance_on_hotpotqa != 'BERT F1 Score: 0.7320' AND application_domain != 'Medical';

-- Query 14: 2 (CSPaper)
SELECT data_modality, uses_knowledge_graph, evaluation_dataset FROM CSPaper WHERE data_modality != 'proprietary enterprise dataset' AND baseline = 'MLP';

-- Query 15: 2 (CSPaper)
SELECT performance_on_hotpotqa, evaluation_metric, baseline FROM CSPaper WHERE baseline != 'BM25' AND reasoning_depth != 'single-hop';

-- Query 16: 2 (CSPaper)
SELECT baseline, uses_reranker, generator_model FROM CSPaper WHERE baseline != 'GPT-3' AND use_agent != 'Yes';

-- Query 17: 2 (CSPaper)
SELECT retrieval_method, baseline, generator_model FROM CSPaper WHERE retrieval_method != 'Other' AND evaluation_metric = 'RMSE';

-- Query 18: 2 (CSPaper)
SELECT multi_turn_retrieval, data_modality, performance_on_NQ FROM CSPaper WHERE data_modality = 'Image' AND evaluation_metric != 'BERTScore';

-- Query 19: 2 (CSPaper)
SELECT performance_on_NQ, data_modality, evaluation_dataset FROM CSPaper WHERE performance_on_NQ != 'Qwen-2.5-7B (Light-BEE): EM 38.71, F1 50.11' AND use_agent = 'Yes';

-- Query 20: 2 (CSPaper)
SELECT generator_model, baseline, performance_on_NQ FROM CSPaper WHERE baseline = 'GPT-3' AND multi_turn_retrieval > 'No';

-- Query 21: 3 (CSPaper)
SELECT retrieval_method, uses_knowledge_graph, multi_turn_retrieval FROM CSPaper WHERE retrieval_method != 'Dense Retrieval' OR evaluation_metric != 'AUC';

-- Query 22: 3 (CSPaper)
SELECT multi_turn_retrieval, baseline, data_modality FROM CSPaper WHERE multi_turn_retrieval >= 'Yes' OR use_agent >= 'Yes';

-- Query 23: 3 (CSPaper)
SELECT uses_knowledge_graph, reasoning_depth, retrieval_method FROM CSPaper WHERE retrieval_method = 'Dense Retrieval' OR performance_on_hotpotqa != 'EM (x4 compression): 0.430';

-- Query 24: 3 (CSPaper)
SELECT performance_on_NQ, uses_knowledge_graph, reasoning_depth FROM CSPaper WHERE reasoning_depth = 'single-hop' OR use_agent >= 'No';

-- Query 25: 3 (CSPaper)
SELECT evaluation_dataset, performance_on_hotpotqa, agent_framework FROM CSPaper WHERE performance_on_hotpotqa = 'F1: 62.87' OR use_agent <= 'No';

-- Query 26: 3 (CSPaper)
SELECT generator_model, uses_reranker, evaluation_metric FROM CSPaper WHERE uses_reranker < 'Yes' OR topic != 'LLM';

-- Query 27: 3 (CSPaper)
SELECT data_modality, multi_turn_retrieval, agent_framework FROM CSPaper WHERE data_modality != 'Image' OR agent_framework = 'ToT';

-- Query 28: 3 (CSPaper)
SELECT performance_on_NQ, reasoning_depth, generator_model FROM CSPaper WHERE reasoning_depth = 'single-hop' OR data_modality != 'proprietary enterprise dataset';

-- Query 29: 3 (CSPaper)
SELECT data_modality, uses_reranker, generator_model FROM CSPaper WHERE data_modality != 'Table' OR evaluation_metric = 'LLM-as-a-judge';

-- Query 30: 3 (CSPaper)
SELECT application_domain, use_agent, reasoning_depth FROM CSPaper WHERE use_agent > 'Yes' OR reasoning_depth = 'single-hop';

-- Query 31: 4 (CSPaper)
SELECT uses_reranker, performance_on_hotpotqa, data_modality FROM CSPaper WHERE performance_on_hotpotqa = 'Comparison F1: 75.30)' AND performance_on_NQ = 'fact_single Recall@5: 0.705' AND use_agent = 'Yes' AND application_domain != 'Sports';

-- Query 32: 4 (CSPaper)
SELECT evaluation_dataset, uses_knowledge_graph, use_agent FROM CSPaper WHERE uses_knowledge_graph > 'single-hop' AND generator_model != 'GPT-3' AND uses_reranker >= 'Text' AND evaluation_dataset != 'Musique';

-- Query 33: 4 (CSPaper)
SELECT performance_on_NQ, data_modality, application_domain FROM CSPaper WHERE data_modality = 'Audio' AND uses_reranker <= 'Yes' AND use_agent < 'No' AND data_modality = 'proprietary enterprise dataset';

-- Query 34: 4 (CSPaper)
SELECT evaluation_metric, generator_model, agent_framework FROM CSPaper WHERE agent_framework >= 'Other' AND use_agent < 'No' AND baseline = 'GPT-3' AND topic != 'Information Retrieval';

-- Query 35: 4 (CSPaper)
SELECT uses_knowledge_graph, baseline, uses_reranker FROM CSPaper WHERE baseline != 'RAPTOR' AND data_modality = 'Audio' AND generator_model = 'Llama3-8b' AND uses_reranker <= 'Text';

-- Query 36: 4 (CSPaper)
SELECT generator_model, retrieval_method, agent_framework FROM CSPaper WHERE generator_model != 'GPT-Neo-1.3B' AND agent_framework >= 'ToT' AND multi_turn_retrieval != 'No' AND topic != 'Retrieval-Augmented Generation';

-- Query 37: 4 (CSPaper)
SELECT uses_reranker, application_domain, evaluation_metric FROM CSPaper WHERE uses_reranker > 'Text' AND performance_on_hotpotqa != 'F1: 0.3066' AND evaluation_dataset != 'WebQSP' AND uses_reranker > 'No';

-- Query 38: 4 (CSPaper)
SELECT topic, application_domain, uses_knowledge_graph FROM CSPaper WHERE topic != 'LLM' AND generator_model != 'Llama3.1-8B-Instruct' AND multi_turn_retrieval < 'Yes' AND topic != 'SFT';

-- Query 39: 4 (CSPaper)
SELECT performance_on_NQ, baseline, evaluation_metric FROM CSPaper WHERE evaluation_metric = 'Contextual Recall' AND agent_framework <= 'ToT' AND evaluation_dataset = 'T-REx' AND evaluation_metric != 'Contextual Recall';

-- Query 40: 4 (CSPaper)
SELECT evaluation_dataset, evaluation_metric, application_domain FROM CSPaper WHERE evaluation_dataset != 'NaturalQuestions' AND uses_knowledge_graph <= 'single-hop' AND multi_turn_retrieval <= 'Yes' AND multi_turn_retrieval = 'No';

-- Query 41: 5 (CSPaper)
SELECT data_modality, agent_framework, generator_model FROM CSPaper WHERE data_modality != 'Image' OR baseline = 'GraphRAG' OR application_domain != 'Sports' OR reasoning_depth != 'multi-hop';

-- Query 42: 5 (CSPaper)
SELECT agent_framework, uses_reranker, use_agent FROM CSPaper WHERE uses_reranker = 'Text' OR retrieval_method != 'Dense Retrieval' OR topic = 'No' OR uses_reranker != 'Text';

-- Query 43: 5 (CSPaper)
SELECT uses_knowledge_graph, retrieval_method, performance_on_NQ FROM CSPaper WHERE uses_knowledge_graph >= 'Yes' OR multi_turn_retrieval = 'No' OR uses_knowledge_graph = 'No' OR evaluation_dataset != 'QuALITY';

-- Query 44: 5 (CSPaper)
SELECT data_modality, uses_reranker, reasoning_depth FROM CSPaper WHERE uses_reranker < 'Text' OR topic != 'Data Selection' OR performance_on_hotpotqa = 'KILT-EM: 27.3 (T5-Base)' OR uses_reranker >= 'Text';

-- Query 45: 5 (CSPaper)
SELECT performance_on_NQ, data_modality, use_agent FROM CSPaper WHERE data_modality != 'proprietary enterprise dataset' OR data_modality != 'Code' OR topic != 'SFT' OR agent_framework < 'ToT';

-- Query 46: 5 (CSPaper)
SELECT use_agent, reasoning_depth, multi_turn_retrieval FROM CSPaper WHERE use_agent < 'No' OR performance_on_hotpotqa != 'EM (x16 compression): 0.426' OR uses_knowledge_graph >= 'single-hop' OR topic = 'LLM';

-- Query 47: 5 (CSPaper)
SELECT topic, performance_on_NQ, performance_on_hotpotqa FROM CSPaper WHERE performance_on_hotpotqa != 'EM: 48.58' OR evaluation_metric = 'Accuracy' OR uses_knowledge_graph < 'Yes' OR data_modality != 'Image';

-- Query 48: 5 (CSPaper)
SELECT evaluation_metric, baseline, evaluation_dataset FROM CSPaper WHERE baseline != 'FLARE' OR uses_knowledge_graph <= 'single-hop' OR performance_on_hotpotqa != 'Acc: 58.4' OR evaluation_metric = 'AUC';

-- Query 49: 5 (CSPaper)
SELECT generator_model, evaluation_metric, agent_framework FROM CSPaper WHERE evaluation_metric = 'F1 score' OR multi_turn_retrieval >= 'No' OR use_agent < 'No' OR topic = 'SFT';

-- Query 50: 5 (CSPaper)
SELECT performance_on_hotpotqa, topic, multi_turn_retrieval FROM CSPaper WHERE multi_turn_retrieval <= 'No' OR performance_on_hotpotqa = 'F1: 62.6' OR reasoning_depth = 'multi-hop' OR performance_on_hotpotqa != 'F1: 62.2';

-- Query 51: 6 (CSPaper)
SELECT retrieval_method, generator_model, reasoning_depth FROM CSPaper WHERE (generator_model = 'Mistral-7B-Instruct-v0.3' AND application_domain = 'Other') OR (agent_framework >= 'CoT' AND performance_on_hotpotqa != 'F1: 59.57');

-- Query 52: 6 (CSPaper)
SELECT evaluation_dataset, generator_model, topic FROM CSPaper WHERE (generator_model != 'LLaMA-2-7B' AND uses_reranker <= 'Text') OR (uses_knowledge_graph >= 'single-hop' AND baseline != 'Vanilla');

-- Query 53: 6 (CSPaper)
SELECT evaluation_dataset, data_modality, agent_framework FROM CSPaper WHERE (data_modality = 'Table' AND performance_on_hotpotqa = 'EM: 42.2') OR (evaluation_dataset != 'MS-MARCO' AND agent_framework >= 'ToT');

-- Query 54: 6 (CSPaper)
SELECT reasoning_depth, topic, multi_turn_retrieval FROM CSPaper WHERE (multi_turn_retrieval != 'No' AND application_domain = 'Education') OR (performance_on_hotpotqa != 'EM: 0.303' AND retrieval_method != 'Sparse Retrieval');

-- Query 55: 6 (CSPaper)
SELECT baseline, use_agent, application_domain FROM CSPaper WHERE (baseline = 'Vanilla RAG' AND performance_on_NQ != 'ROUGE-L: 0.0322') OR (performance_on_NQ = 'Llama-3.1: 67%' AND evaluation_metric = 'F1 score');

-- Query 56: 6 (CSPaper)
SELECT reasoning_depth, baseline, multi_turn_retrieval FROM CSPaper WHERE (baseline != 'FL-RAG' AND retrieval_method = 'Other') OR (baseline = 'PoisonedRAG' AND baseline != 'PoisonedRAG');

-- Query 57: 6 (CSPaper)
SELECT performance_on_hotpotqa, uses_reranker, evaluation_dataset FROM CSPaper WHERE (performance_on_hotpotqa != 'Compression of top 5 documents: EM: 33.67, F1: 45.06' AND uses_reranker < 'No') OR (application_domain = 'Sports' AND uses_knowledge_graph < 'single-hop');

-- Query 58: 6 (CSPaper)
SELECT reasoning_depth, evaluation_metric, application_domain FROM CSPaper WHERE (reasoning_depth != 'single-hop' AND multi_turn_retrieval != 'Yes') OR (performance_on_NQ = 'Mistral-I7B (Keyword): acc: 61.0, cacc: 45.0' AND performance_on_NQ = 'LLaMA-3-8B (Light-BEE): EM 38.42, F1 49.23');

-- Query 59: 6 (CSPaper)
SELECT application_domain, topic, performance_on_hotpotqa FROM CSPaper WHERE (performance_on_hotpotqa != 'Acc: 55.68' AND evaluation_metric = 'Recall') OR (multi_turn_retrieval > 'Yes' AND uses_knowledge_graph <= 'single-hop');

-- Query 60: 6 (CSPaper)
SELECT application_domain, performance_on_hotpotqa, retrieval_method FROM CSPaper WHERE (performance_on_hotpotqa = 'F1: 0.3066' AND uses_knowledge_graph >= 'single-hop') OR (agent_framework > 'ToT' AND baseline != 'GPT-4');

