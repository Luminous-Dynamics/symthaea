{{/*
Expand the name of the chart.
*/}}
{{- define "mycelix.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mycelix.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mycelix.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mycelix.labels" -}}
helm.sh/chart: {{ include "mycelix.chart" . }}
{{ include "mycelix.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mycelix.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mycelix.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API labels
*/}}
{{- define "mycelix.api.labels" -}}
{{ include "mycelix.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "mycelix.api.selectorLabels" -}}
{{ include "mycelix.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Web labels
*/}}
{{- define "mycelix.web.labels" -}}
{{ include "mycelix.labels" . }}
app.kubernetes.io/component: web
{{- end }}

{{/*
Web selector labels
*/}}
{{- define "mycelix.web.selectorLabels" -}}
{{ include "mycelix.selectorLabels" . }}
app.kubernetes.io/component: web
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mycelix.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mycelix.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database URL
*/}}
{{- define "mycelix.databaseUrl" -}}
{{- if .Values.api.secrets.databaseUrl }}
{{- .Values.api.secrets.databaseUrl }}
{{- else if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s-postgresql:5432/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password (include "mycelix.fullname" .) .Values.postgresql.auth.database }}
{{- else }}
{{- fail "Either api.secrets.databaseUrl or postgresql.enabled must be set" }}
{{- end }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "mycelix.redisUrl" -}}
{{- if .Values.api.secrets.redisUrl }}
{{- .Values.api.secrets.redisUrl }}
{{- else if .Values.redis.enabled }}
{{- printf "redis://%s-redis-master:6379" (include "mycelix.fullname" .) }}
{{- else }}
{{- fail "Either api.secrets.redisUrl or redis.enabled must be set" }}
{{- end }}
{{- end }}
