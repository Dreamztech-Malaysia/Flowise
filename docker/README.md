# Flowise Docker Hub Image

Starts Flowise from [DockerHub Image](https://hub.docker.com/r/flowiseai/flowise)

## Usage

1. Create `.env` file and specify the `PORT` (refer to `.env.example`)
2. `docker compose up -d`
3. Open [http://localhost:3000](http://localhost:3000)
4. You can bring the containers down by `docker compose stop`

## 🌱 Env Variables

If you like to persist your data (flows, logs, credentials, storage), set these variables in the `.env` file inside `docker` folder:

-   DATABASE_PATH=/home/node/.flowise
-   LOG_PATH=/home/node/.flowise/logs
-   SECRETKEY_PATH=/home/node/.flowise
-   BLOB_STORAGE_PATH=/home/node/.flowise/storage

Flowise also support different environment variables to configure your instance. Read [more](https://docs.flowiseai.com/configuration/environment-variables)

> The container runs as the non-root `node` user (uid 1000), whose home directory is `/home/node`. If you bind-mount a host directory (e.g. `~/.flowise`) for persistence, make sure it's writable by that user - on Linux hosts this may require `chown -R 1000:1000 ~/.flowise`.

### Customer memory (Mem0 self-hosted)

The Mem0 memory node's **Self-hosted** mode stores customer facts in Qdrant. Set in `.env`:

-   `QDRANT_URL` (e.g. `http://qdrant:6333`), `QDRANT_API_KEY` (if the Qdrant server requires one)
-   optional: `CUSTOMER_MEMORY_COLLECTION` (default `customer_memory`), `CUSTOMER_MEMORY_EMBEDDING_DIMS` (default `1536`), `MEM0_TELEMETRY` (defaults to `false`)

No LLM/embedding key is configured here: each request supplies the customer's own key in `overrideConfig.customerMemory`.
Read/delete facts via `/api/v1/customer-memory` (Flowise API key auth).

## Queue Mode:

### Building from source:

You can build the images for worker and main from scratch with:

```
docker compose -f docker-compose-queue-source.yml up -d
```

Monitor Health:

```
docker compose -f docker-compose-queue-source.yml ps
```

### From pre-built images:

You can also use the pre-built images:

```
docker compose -f docker-compose-queue-prebuilt.yml up -d
```

Monitor Health:

```
docker compose -f docker-compose-queue-prebuilt.yml ps
```
