-- Query 1: filter1_agg1 (cspaper)
SELECT uses_knowledge_graph, COUNT(evaluation_metric) AS count_evaluation_metric FROM cspaper WHERE reasoning_depth = 'multi-hop' GROUP BY uses_knowledge_graph;

-- Query 2: filter2_agg1 (cspaper)
SELECT reasoning_depth, SUM(baseline_amount) AS sum_baseline_amount FROM cspaper WHERE retrieval_method = 'Web Search; Dense Retrieval' AND use_agent = 'No' GROUP BY reasoning_depth;

-- Query 3: filter3_agg1 (cspaper)
SELECT data_modality, AVG(baseline_amount) AS avg_baseline_amount FROM cspaper WHERE application_domain != 'Education' OR use_agent = 'Yes' GROUP BY data_modality;

-- Query 4: filter4_agg1 (cspaper)
SELECT data_modality, AVG(baseline_amount) AS avg_baseline_amount FROM cspaper WHERE use_agent = 'Yes' AND retrieval_method != 'Web Search; Dense Retrieval' AND data_modality != 'Table; Text' GROUP BY data_modality;

-- Query 5: filter5_agg1 (cspaper)
SELECT retrieval_method, SUM(baseline_amount) AS sum_baseline_amount FROM cspaper WHERE uses_reranker = 'No' OR performance_on_hotpotqa = 'KILT-EM: 27.3 (T5-Base); KILT-EM: 31.1 (T5-XL)' OR baseline = 'Traditional RAG' GROUP BY retrieval_method;

-- Query 6: filter6_agg1 (cspaper)
SELECT uses_knowledge_graph, COUNT(reasoning_depth) AS count_reasoning_depth FROM cspaper WHERE (baseline = 'e5-base-v2; gpt-2' AND topic = 'SFT') OR (data_modality = 'Image; Table; Text' AND topic != 'SFT') GROUP BY uses_knowledge_graph;

