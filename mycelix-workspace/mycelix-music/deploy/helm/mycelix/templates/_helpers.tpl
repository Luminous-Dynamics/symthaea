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
Database host
*/}}
{{- define "mycelix.databaseHost" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "mycelix.fullname" .) }}
{{- else }}
{{- .Values.externalDatabase.host }}
{{- end }}
{{- end }}

{{/*
Database port
*/}}
{{- define "mycelix.databasePort" -}}
{{- if .Values.postgresql.enabled }}
{{- default "5432" .Values.postgresql.primary.service.ports.postgresql }}
{{- else }}
{{- default "5432" .Values.externalDatabase.port }}
{{- end }}
{{- end }}

{{/*
Redis host
*/}}
{{- define "mycelix.redisHost" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "mycelix.fullname" .) }}
{{- else }}
{{- .Values.externalRedis.host }}
{{- end }}
{{- end }}

{{/*
Elasticsearch host
*/}}
{{- define "mycelix.elasticsearchHost" -}}
{{- if .Values.elasticsearch.enabled }}
{{- printf "http://%s-elasticsearch:9200" (include "mycelix.fullname" .) }}
{{- else }}
{{- .Values.externalElasticsearch.host }}
{{- end }}
{{- end }}

{{/*
Kafka brokers
*/}}
{{- define "mycelix.kafkaBrokers" -}}
{{- if .Values.kafka.enabled }}
{{- printf "%s-kafka:9092" (include "mycelix.fullname" .) }}
{{- else }}
{{- .Values.externalKafka.brokers }}
{{- end }}
{{- end }}
