-- Query 1: aggregation (cspaper)
SELECT agent_framework, MIN(data_modality) AS min_data_modality FROM cspaper GROUP BY agent_framework;

-- Query 2: aggregation (cspaper)
SELECT uses_knowledge_graph, COUNT(topic) AS count_topic FROM cspaper GROUP BY uses_knowledge_graph;

-- Query 3: aggregation (cspaper)
SELECT uses_reranker, AVG(data_modality) AS avg_data_modality FROM cspaper GROUP BY uses_reranker;

-- Query 4: aggregation (cspaper)
SELECT topic, MAX(data_modality) AS max_data_modality FROM cspaper GROUP BY topic;

-- Query 5: aggregation (cspaper)
SELECT uses_knowledge_graph, COUNT(use_agent) AS count_use_agent FROM cspaper GROUP BY uses_knowledge_graph;

-- Query 6: aggregation (cspaper)
SELECT agent_framework, MIN(data_modality) AS min_data_modality FROM cspaper GROUP BY agent_framework;

-- Query 7: aggregation (cspaper)
SELECT topic, MAX(data_modality) AS max_data_modality FROM cspaper GROUP BY topic;

-- Query 8: aggregation (cspaper)
SELECT reasoning_depth, SUM(data_modality) AS sum_data_modality FROM cspaper GROUP BY reasoning_depth;

