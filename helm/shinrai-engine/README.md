# shinrai-engine Helm chart

Installs the ShinrAI Engine — the self-hostable HTTP inference service for
the ShinrAI PII detection models. `helm install` with defaults works on a
stock cluster: CPU image, fp32, model downloaded from Hugging Face on first
start (~1.3 GB) into an emptyDir.

```bash
helm install shinrai-engine ./helm/shinrai-engine \
  --namespace shinrai-engine --create-namespace \
  --set persistence.enabled=true \
  --set apiKey.value="$(openssl rand -hex 24)"
```

## microk8s

```bash
microk8s enable hostpath-storage    # persistence.enabled=true uses the default SC
microk8s enable gpu                 # only for gpu.enabled=true (NVIDIA operator)
microk8s enable ingress             # only for ingress.enabled=true (className: public)
```

## The knobs that matter

| Value | Why |
|---|---|
| `persistence.enabled` | Off = every pod restart re-downloads the model. Turn it on. |
| `apiKey.existingSecret` / `apiKey.value` | Off = `/api/*` is open to the whole cluster. |
| `model.precision` | `fp32` is the only supported production precision. `q8`/`int4` print an honest warning into the release notes and onto `/healthz`. |
| `gpu.enabled` | Adds `nvidia.com/gpu: 1` and switches to the `-gpu` image tag. Check `/healthz` `providers` afterwards — CUDA support depends on your card (see the repo README, GPU section). |
| `model.hfTokenSecret` | Only needed for private model repos. |

Full documentation: https://github.com/Innovius-ai/shinrai-engine
