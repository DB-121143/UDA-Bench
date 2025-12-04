## UQE Usage

### Basic Usage

```bash
# View help
python main.py --help

# Run dataset with custom query type
python main.py --dataset lcr --query-type SFW
```

### Data and Query Files
Data should be saved in JSON format. Text is directly written to JSON files, while other multimodal data saves paths. Column names can be modified as needed.
```json
{
    "id": "1",
    "description": "your-text",
    "image": "your-path-to-image"
}
```
Queries should be saved as `query/{dataset}/{query-type}/x.sql`


### Command Line Arguments

- `--dataset, -d`: Specify the dataset to run
- `--all, -a`: Run all datasets
- `--query-type, -q`: Override query type

### Configuration File config_uqe.py
Main UQE configuration parameters:
- `USE_BART`: Whether to use BART model
- `BATCH_SIZE`: Batch size
- `BUDGET`: Budget limit
- `AGGR_STRATEGY`: Aggregation strategy
- `N_CENTROIDS`: Number of clustering centroids
- `N_ITER`: Number of iterations
- `GROUP_EXTRACT_SAMPLE_RATIO`: Group extraction sampling ratio
- `AGGR_CLUSTER_SAMPLE_RATIO`: Aggregation cluster sampling ratio

OpenAI connection configuration:
- `MODEL`: Selected OpenAI model
- `OPENAI_KEY`: OpenAI API key
- `BASE_URL`: OpenAI base URL

## Output Results

- Results are saved in `result/{dataset_name}/{query_type}/{timestamp}/` directory
- Each query result is saved as a CSV file
