
# Kafka Integration Setup

## Environment variables

```bash
export KAFKA_BOOTSTRAP=localhost:9092
export KAFKA_TOPIC=events.raw
export KAFKA_GROUP_ID=anomaly-detector-1
Planned features
□ Consumer group with rebalance callback
□ Offset commit every N events
□ Dead letter queue for malformed events
□ Metrics: lag, throughput, error rate
