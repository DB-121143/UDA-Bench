-- Query 1: aggregation (CSPaper)
SELECT agent_framework, MIN(data_modality) AS min_data_modality FROM CSPaper GROUP BY agent_framework;

-- Query 2: aggregation (CSPaper)
SELECT uses_knowledge_graph, COUNT(topic) AS count_topic FROM CSPaper GROUP BY uses_knowledge_graph;

-- Query 3: aggregation (CSPaper)
SELECT uses_reranker, AVG(data_modality) AS avg_data_modality FROM CSPaper GROUP BY uses_reranker;

-- Query 4: aggregation (CSPaper)
SELECT topic, MAX(data_modality) AS max_data_modality FROM CSPaper GROUP BY topic;

-- Query 5: aggregation (CSPaper)
SELECT uses_knowledge_graph, COUNT(use_agent) AS count_use_agent FROM CSPaper GROUP BY uses_knowledge_graph;

-- Query 6: aggregation (CSPaper)
SELECT agent_framework, MIN(data_modality) AS min_data_modality FROM CSPaper GROUP BY agent_framework;

-- Query 7: aggregation (CSPaper)
SELECT topic, MAX(data_modality) AS max_data_modality FROM CSPaper GROUP BY topic;

-- Query 8: aggregation (CSPaper)
SELECT reasoning_depth, SUM(data_modality) AS sum_data_modality FROM CSPaper GROUP BY reasoning_depth;

