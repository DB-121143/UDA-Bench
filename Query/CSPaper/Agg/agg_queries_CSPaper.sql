-- Query 1: aggregation (cspaper)
SELECT use_agent, MIN(baseline_amount) AS min_baseline_amount FROM cspaper GROUP BY use_agent;

-- Query 2: aggregation (cspaper)
SELECT uses_knowledge_graph, COUNT(topic) AS count_topic FROM cspaper GROUP BY uses_knowledge_graph;

-- Query 3: aggregation (cspaper)
SELECT uses_reranker, AVG(baseline_amount) AS avg_baseline_amount FROM cspaper GROUP BY uses_reranker;

-- Query 4: aggregation (cspaper)
SELECT topic, MAX(baseline_amount) AS max_baseline_amount FROM cspaper GROUP BY topic;

-- Query 5: aggregation (cspaper)
SELECT uses_knowledge_graph, COUNT(baseline_amount) AS count_baseline_amount FROM cspaper GROUP BY uses_knowledge_graph;

-- Query 6: aggregation (cspaper)
SELECT use_agent, MIN(baseline_amount) AS min_baseline_amount FROM cspaper GROUP BY use_agent;

-- Query 7: aggregation (cspaper)
SELECT topic, MAX(baseline_amount) AS max_baseline_amount FROM cspaper GROUP BY topic;

-- Query 8: aggregation (cspaper)
SELECT reasoning_depth, SUM(baseline_amount) AS sum_baseline_amount FROM cspaper GROUP BY reasoning_depth;

