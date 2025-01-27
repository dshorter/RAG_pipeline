```sql
-- Core flattened metrics view
CREATE VIEW vw_powerbi_metrics AS
SELECT 
    m.metric_id,
    m.timestamp,
    m.operation,
    m.component,
    m.success,
    m.chunk_id,
    m.document_id,
    
    -- Extract common performance metrics
    JSON_VALUE(m.metrics_data, '$.duration_ms') as duration_ms,
    JSON_VALUE(m.metrics_data, '$.token_count') as token_count,
    JSON_VALUE(m.metrics_data, '$.memory_used_mb') as memory_used_mb,
    
    -- Model metrics
    JSON_VALUE(m.metrics_data, '$.model.name') as model_name,
    JSON_VALUE(m.metrics_data, '$.model.dimensions') as model_dimensions,
    
    -- Quality metrics
    JSON_VALUE(m.metrics_data, '$.relevance_score') as relevance_score,
    JSON_VALUE(m.metrics_data, '$.citation_count') as citation_count
FROM metrics m;

-- Hourly aggregation view
CREATE VIEW vw_hourly_metrics AS
SELECT
    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
    operation,
    component,
    COUNT(*) as operation_count,
    AVG(CAST(JSON_VALUE(metrics_data, '$.duration_ms') AS FLOAT)) as avg_duration_ms,
    SUM(success) * 100.0 / COUNT(*) as success_rate,
    AVG(CAST(JSON_VALUE(metrics_data, '$.memory_used_mb') AS FLOAT)) as avg_memory_mb
FROM metrics
GROUP BY 
    strftime('%Y-%m-%d %H:00:00', timestamp),
    operation,
    component;

-- Model performance view
CREATE VIEW vw_model_metrics AS
SELECT
    strftime('%Y-%m-%d', timestamp) as date,
    JSON_VALUE(metrics_data, '$.model.name') as model_name,
    COUNT(*) as operation_count,
    AVG(CAST(JSON_VALUE(metrics_data, '$.duration_ms') AS FLOAT)) as avg_duration_ms,
    AVG(CAST(JSON_VALUE(metrics_data, '$.relevance_score') AS FLOAT)) as avg_relevance_score,
    SUM(success) * 100.0 / COUNT(*) as success_rate
FROM metrics
WHERE JSON_VALUE(metrics_data, '$.model.name') IS NOT NULL
GROUP BY 
    strftime('%Y-%m-%d', timestamp),
    JSON_VALUE(metrics_data, '$.model.name');
```
