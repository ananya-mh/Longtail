#!/usr/bin/env bash
# Deploy Longtail to this team's K8s namespace at http://<team-host>/app/
# Pattern from the challenge's deploy-app-no-registry skill: public python image, code from a
# ConfigMap, credentials from a Secret, Ingress path /app. Run on the workshop VM: ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
export KUBECONFIG=/config/kubeconfig

mapfile -t TEAM_CONFIGS < <(find /config -maxdepth 1 -type f -name '*.config' | sort)
(( ${#TEAM_CONFIGS[@]} == 1 )) || { echo "expected exactly one /config/*.config"; exit 1; }
set -a && source "${TEAM_CONFIGS[0]}" && set +a

NS="$USERNAME"
APP=longtail
PORT=8080
HOST="${INGRESS_URL#http://}"; HOST="${HOST#https://}"; HOST="${HOST%%/*}"

kubectl -n "$NS" create configmap "${APP}-code" --from-file=app --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" create secret generic "${APP}-creds" \
  --from-literal=VSS_URL="$INGRESS_URL" \
  --from-literal=VSS_USERNAME="$USERNAME" \
  --from-literal=VSS_PASSWORD="$PASSWORD" \
  --from-literal=GPU_BEARER_TOKEN="${GPU_BEARER_TOKEN:-}" \
  --from-literal=WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --from-literal=WANDB_TEAM="${WANDB_TEAM:-}" \
  --from-literal=WANDB_PROJECT="${WANDB_PROJECT:-longtail}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata: {name: ${APP}, labels: {app: ${APP}}}
spec:
  replicas: 1
  selector: {matchLabels: {app: ${APP}}}
  template:
    metadata: {labels: {app: ${APP}}}
    spec:
      containers:
      - name: app
        image: python:3.12-slim
        ports: [{containerPort: ${PORT}}]
        env:
        - {name: PORT, value: "${PORT}"}
        - {name: AGENT_MODEL, value: "${AGENT_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B}"}
        envFrom: [{secretRef: {name: ${APP}-creds}}]
        volumeMounts: [{name: code, mountPath: /code}]
        workingDir: /code
        command: ["bash", "-c"]
        args: ["pip install --no-cache-dir -q -r requirements.txt && exec python main.py"]
        readinessProbe:
          httpGet: {path: /health, port: ${PORT}}
          initialDelaySeconds: 20
          periodSeconds: 10
      volumes: [{name: code, configMap: {name: ${APP}-code}}]
---
apiVersion: v1
kind: Service
metadata: {name: ${APP}, labels: {app: ${APP}}}
spec:
  selector: {app: ${APP}}
  ports: [{name: http, port: 80, targetPort: ${PORT}}]
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${APP}
  labels: {app: ${APP}}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  ingressClassName: nginx
  rules:
  - host: ${HOST}
    http:
      paths:
      - path: /app(/|$)(.*)
        pathType: ImplementationSpecific
        backend: {service: {name: ${APP}, port: {number: 80}}}
YAML

kubectl -n "$NS" rollout restart deploy/"$APP"
kubectl -n "$NS" rollout status deploy/"$APP" --timeout=240s
echo "Live at: http://${HOST}/app/"
