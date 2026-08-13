{{- define "shinrai-engine.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "shinrai-engine.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "shinrai-engine.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
app.kubernetes.io/name: {{ include "shinrai-engine.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "shinrai-engine.selectorLabels" -}}
app.kubernetes.io/name: {{ include "shinrai-engine.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "shinrai-engine.imageTag" -}}
{{- /* CI publishes v-prefixed tags (v0.1.2), never the bare appVersion —
       and gpu.enabled must keep its -gpu suffix even when a tag is pinned,
       or a pinned GPU deployment silently runs the CPU-only image. */ -}}
{{- $base := .Values.image.tag | default (printf "v%s" .Chart.AppVersion) -}}
{{- $base = trimSuffix "-gpu" $base -}}
{{- if .Values.gpu.enabled -}}
{{- printf "%s-gpu" $base -}}
{{- else -}}
{{- $base -}}
{{- end -}}
{{- end -}}

{{- define "shinrai-engine.apiKeySecretName" -}}
{{- if .Values.apiKey.existingSecret -}}
{{- .Values.apiKey.existingSecret -}}
{{- else -}}
{{- printf "%s-api-key" (include "shinrai-engine.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "shinrai-engine.authEnabled" -}}
{{- if or .Values.apiKey.existingSecret .Values.apiKey.value -}}true{{- end -}}
{{- end -}}
