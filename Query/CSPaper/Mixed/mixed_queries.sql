-- Query 1: filter1_agg1 (cspaper)
SELECT uses_reranker, AVG(data_modality) AS avg_data_modality FROM cspaper WHERE evaluation_metric != 'F1' GROUP BY uses_reranker;

-- Query 2: filter2_agg1 (cspaper)
SELECT topic, COUNT(agent_framework) AS count_agent_framework FROM cspaper WHERE topic != 'Retrieval-Augmented Generation' AND uses_reranker > 'Yes' GROUP BY topic;

-- Query 3: filter3_agg1 (cspaper)
SELECT agent_framework, MAX(data_modality) AS max_data_modality FROM cspaper WHERE performance_on_hotpotqa = 'EM: 43.75 (with Llama-3.1-8B-Instruct)' OR agent_framework >= 'Other' GROUP BY agent_framework;

-- Query 4: filter4_agg1 (cspaper)
SELECT use_agent, MAX(data_modality) AS max_data_modality FROM cspaper WHERE application_domain != 'Art' AND evaluation_metric = 'FactScore' AND uses_knowledge_graph > 'Yes' GROUP BY use_agent;

-- Query 5: filter5_agg1 (cspaper)
SELECT retrieval_method, SUM(data_modality) AS sum_data_modality FROM cspaper WHERE data_modality = 'Table' OR evaluation_metric != 'Coherence' OR topic = 'Information Retrieval' GROUP BY retrieval_method;

-- Query 6: filter6_agg1 (cspaper)
SELECT uses_reranker, MAX(data_modality) AS max_data_modality FROM cspaper WHERE (evaluation_dataset != 'InfoSeek' AND data_modality = 'proprietary enterprise dataset') OR (baseline != 'No RAG' AND baseline != 'PoisonedRAG') GROUP BY uses_reranker;

