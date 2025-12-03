-- Query 1: aggregation (CSPaper_paper)
SELECT agent_framework, MIN(data_modality) AS min_data_modality FROM CSPaper_paper GROUP BY agent_framework;

-- Query 2: aggregation (CSPaper_paper)
SELECT uses_knowledge_graph, COUNT(topic) AS count_topic FROM CSPaper_paper GROUP BY uses_knowledge_graph;

-- Query 3: aggregation (CSPaper_paper)
SELECT uses_reranker, AVG(data_modality) AS avg_data_modality FROM CSPaper_paper GROUP BY uses_reranker;

-- Query 4: aggregation (CSPaper_paper)
SELECT topic, MAX(data_modality) AS max_data_modality FROM CSPaper_paper GROUP BY topic;

-- Query 5: aggregation (CSPaper_paper)
SELECT uses_knowledge_graph, COUNT(use_agent) AS count_use_agent FROM CSPaper_paper GROUP BY uses_knowledge_graph;

-- Query 6: aggregation (CSPaper_paper)
SELECT agent_framework, MIN(data_modality) AS min_data_modality FROM CSPaper_paper GROUP BY agent_framework;

-- Query 7: aggregation (CSPaper_paper)
SELECT topic, MAX(data_modality) AS max_data_modality FROM CSPaper_paper GROUP BY topic;

-- Query 8: aggregation (CSPaper_paper)
SELECT reasoning_depth, SUM(data_modality) AS sum_data_modality FROM CSPaper_paper GROUP BY reasoning_depth;

-- Query 9: aggregation (CSPaper_paper)
SELECT topic, COUNT(uses_reranker) AS count_uses_reranker FROM CSPaper_paper GROUP BY topic;

-- Query 10: aggregation (CSPaper_paper)
SELECT topic, COUNT(evaluation_metric) AS count_evaluation_metric FROM CSPaper_paper GROUP BY topic;

