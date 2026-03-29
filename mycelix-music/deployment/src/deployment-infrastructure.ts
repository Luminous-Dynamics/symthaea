// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Deployment & Multi-Region Infrastructure
 *
 * Complete deployment infrastructure including:
 * - Infrastructure as Code (Terraform/Pulumi style)
 * - CI/CD Pipeline Definitions
 * - Multi-Region Architecture
 * - Feature Flags System
 * - GitOps Configuration
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';

// ============================================================================
// Infrastructure as Code
// ============================================================================

interface IaCResource {
  type: string;
  name: string;
  provider: string;
  properties: Record<string, any>;
  dependencies?: string[];
  outputs?: Record<string, string>;
}

interface IaCStack {
  name: string;
  environment: string;
  region: string;
  resources: IaCResource[];
  variables: Record<string, any>;
  outputs: Record<string, any>;
}

interface IaCPlan {
  stackName: string;
  additions: IaCResource[];
  modifications: Array<{ resource: IaCResource; changes: Record<string, { old: any; new: any }> }>;
  deletions: IaCResource[];
  unchanged: IaCResource[];
}

export class InfrastructureAsCode {
  private stacks: Map<string, IaCStack> = new Map();
  private providers: Map<string, IaCProvider> = new Map();

  constructor() {
    this.registerDefaultProviders();
  }

  private registerDefaultProviders(): void {
    this.providers.set('aws', new AWSProvider());
    this.providers.set('gcp', new GCPProvider());
    this.providers.set('kubernetes', new KubernetesProvider());
  }

  defineStack(config: Omit<IaCStack, 'resources' | 'outputs'>): StackBuilder {
    const stack: IaCStack = {
      ...config,
      resources: [],
      outputs: {},
    };
    this.stacks.set(stack.name, stack);
    return new StackBuilder(stack, this.providers);
  }

  async plan(stackName: string): Promise<IaCPlan> {
    const stack = this.stacks.get(stackName);
    if (!stack) throw new Error(`Stack ${stackName} not found`);

    // Compare with current state (would fetch from state storage)
    const currentState = await this.getCurrentState(stackName);

    const plan: IaCPlan = {
      stackName,
      additions: [],
      modifications: [],
      deletions: [],
      unchanged: [],
    };

    // Find additions and modifications
    for (const resource of stack.resources) {
      const existing = currentState.find(r => r.name === resource.name && r.type === resource.type);

      if (!existing) {
        plan.additions.push(resource);
      } else {
        const changes = this.diffResources(existing, resource);
        if (Object.keys(changes).length > 0) {
          plan.modifications.push({ resource, changes });
        } else {
          plan.unchanged.push(resource);
        }
      }
    }

    // Find deletions
    for (const existing of currentState) {
      const stillExists = stack.resources.find(r => r.name === existing.name && r.type === existing.type);
      if (!stillExists) {
        plan.deletions.push(existing);
      }
    }

    return plan;
  }

  async apply(stackName: string): Promise<void> {
    const plan = await this.plan(stackName);
    const stack = this.stacks.get(stackName)!;

    console.log(`Applying changes to ${stackName}...`);

    // Delete removed resources
    for (const resource of plan.deletions) {
      await this.deleteResource(resource);
    }

    // Create new resources
    for (const resource of plan.additions) {
      await this.createResource(resource);
    }

    // Update modified resources
    for (const { resource } of plan.modifications) {
      await this.updateResource(resource);
    }

    // Save state
    await this.saveState(stackName, stack.resources);
  }

  private async getCurrentState(stackName: string): Promise<IaCResource[]> {
    // Would fetch from remote state (S3, GCS, etc.)
    return [];
  }

  private async saveState(stackName: string, resources: IaCResource[]): Promise<void> {
    // Would save to remote state storage
    console.log(`State saved for ${stackName}`);
  }

  private diffResources(a: IaCResource, b: IaCResource): Record<string, { old: any; new: any }> {
    const changes: Record<string, { old: any; new: any }> = {};

    for (const key of Object.keys(b.properties)) {
      if (JSON.stringify(a.properties[key]) !== JSON.stringify(b.properties[key])) {
        changes[key] = { old: a.properties[key], new: b.properties[key] };
      }
    }

    return changes;
  }

  private async createResource(resource: IaCResource): Promise<void> {
    const provider = this.providers.get(resource.provider);
    if (provider) {
      await provider.create(resource);
    }
  }

  private async updateResource(resource: IaCResource): Promise<void> {
    const provider = this.providers.get(resource.provider);
    if (provider) {
      await provider.update(resource);
    }
  }

  private async deleteResource(resource: IaCResource): Promise<void> {
    const provider = this.providers.get(resource.provider);
    if (provider) {
      await provider.delete(resource);
    }
  }

  generateTerraformHCL(stackName: string): string {
    const stack = this.stacks.get(stackName);
    if (!stack) throw new Error(`Stack ${stackName} not found`);

    let hcl = `# Generated Terraform configuration for ${stackName}\n\n`;

    // Provider configuration
    hcl += `terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  backend "s3" {
    bucket = "mycelix-terraform-state"
    key    = "${stackName}/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "${stack.region}"
}

`;

    // Variables
    hcl += `# Variables\n`;
    for (const [name, value] of Object.entries(stack.variables)) {
      hcl += `variable "${name}" {
  default = ${JSON.stringify(value)}
}\n\n`;
    }

    // Resources
    for (const resource of stack.resources) {
      hcl += this.resourceToHCL(resource);
    }

    // Outputs
    hcl += `# Outputs\n`;
    for (const [name, value] of Object.entries(stack.outputs)) {
      hcl += `output "${name}" {
  value = ${value}
}\n\n`;
    }

    return hcl;
  }

  private resourceToHCL(resource: IaCResource): string {
    let hcl = `resource "${resource.type}" "${resource.name}" {\n`;

    for (const [key, value] of Object.entries(resource.properties)) {
      if (typeof value === 'object' && !Array.isArray(value)) {
        hcl += `  ${key} {\n`;
        for (const [k, v] of Object.entries(value as Record<string, any>)) {
          hcl += `    ${k} = ${JSON.stringify(v)}\n`;
        }
        hcl += `  }\n`;
      } else {
        hcl += `  ${key} = ${JSON.stringify(value)}\n`;
      }
    }

    if (resource.dependencies?.length) {
      hcl += `  depends_on = [${resource.dependencies.map(d => `"${d}"`).join(', ')}]\n`;
    }

    hcl += `}\n\n`;
    return hcl;
  }
}

class StackBuilder {
  constructor(private stack: IaCStack, private providers: Map<string, IaCProvider>) {}

  addVPC(config: {
    name: string;
    cidr: string;
    azs: string[];
    publicSubnets: string[];
    privateSubnets: string[];
  }): this {
    this.stack.resources.push({
      type: 'aws_vpc',
      name: config.name,
      provider: 'aws',
      properties: {
        cidr_block: config.cidr,
        enable_dns_hostnames: true,
        enable_dns_support: true,
        tags: { Name: config.name, Environment: this.stack.environment },
      },
    });

    // Add subnets
    config.azs.forEach((az, i) => {
      if (config.publicSubnets[i]) {
        this.stack.resources.push({
          type: 'aws_subnet',
          name: `${config.name}-public-${az}`,
          provider: 'aws',
          properties: {
            vpc_id: `\${aws_vpc.${config.name}.id}`,
            cidr_block: config.publicSubnets[i],
            availability_zone: az,
            map_public_ip_on_launch: true,
          },
          dependencies: [`aws_vpc.${config.name}`],
        });
      }
      if (config.privateSubnets[i]) {
        this.stack.resources.push({
          type: 'aws_subnet',
          name: `${config.name}-private-${az}`,
          provider: 'aws',
          properties: {
            vpc_id: `\${aws_vpc.${config.name}.id}`,
            cidr_block: config.privateSubnets[i],
            availability_zone: az,
          },
          dependencies: [`aws_vpc.${config.name}`],
        });
      }
    });

    return this;
  }

  addEKSCluster(config: {
    name: string;
    vpcName: string;
    version: string;
    nodeGroups: Array<{
      name: string;
      instanceTypes: string[];
      minSize: number;
      maxSize: number;
      desiredSize: number;
    }>;
  }): this {
    this.stack.resources.push({
      type: 'aws_eks_cluster',
      name: config.name,
      provider: 'aws',
      properties: {
        name: config.name,
        version: config.version,
        role_arn: `\${aws_iam_role.${config.name}-cluster-role.arn}`,
        vpc_config: {
          subnet_ids: `\${aws_subnet.${config.vpcName}-private-*.id}`,
        },
      },
    });

    config.nodeGroups.forEach(ng => {
      this.stack.resources.push({
        type: 'aws_eks_node_group',
        name: `${config.name}-${ng.name}`,
        provider: 'aws',
        properties: {
          cluster_name: config.name,
          node_group_name: ng.name,
          node_role_arn: `\${aws_iam_role.${config.name}-node-role.arn}`,
          subnet_ids: `\${aws_subnet.${config.vpcName}-private-*.id}`,
          instance_types: ng.instanceTypes,
          scaling_config: {
            min_size: ng.minSize,
            max_size: ng.maxSize,
            desired_size: ng.desiredSize,
          },
        },
        dependencies: [`aws_eks_cluster.${config.name}`],
      });
    });

    return this;
  }

  addRDS(config: {
    name: string;
    engine: string;
    engineVersion: string;
    instanceClass: string;
    allocatedStorage: number;
    multiAZ: boolean;
    readReplicas?: number;
  }): this {
    this.stack.resources.push({
      type: 'aws_db_instance',
      name: config.name,
      provider: 'aws',
      properties: {
        identifier: config.name,
        engine: config.engine,
        engine_version: config.engineVersion,
        instance_class: config.instanceClass,
        allocated_storage: config.allocatedStorage,
        multi_az: config.multiAZ,
        storage_encrypted: true,
        backup_retention_period: 7,
        skip_final_snapshot: false,
        final_snapshot_identifier: `${config.name}-final`,
      },
    });

    if (config.readReplicas) {
      for (let i = 0; i < config.readReplicas; i++) {
        this.stack.resources.push({
          type: 'aws_db_instance',
          name: `${config.name}-replica-${i}`,
          provider: 'aws',
          properties: {
            identifier: `${config.name}-replica-${i}`,
            replicate_source_db: `\${aws_db_instance.${config.name}.id}`,
            instance_class: config.instanceClass,
          },
          dependencies: [`aws_db_instance.${config.name}`],
        });
      }
    }

    return this;
  }

  addElastiCache(config: {
    name: string;
    engine: 'redis' | 'memcached';
    nodeType: string;
    numNodes: number;
    clusterMode?: boolean;
  }): this {
    this.stack.resources.push({
      type: 'aws_elasticache_replication_group',
      name: config.name,
      provider: 'aws',
      properties: {
        replication_group_id: config.name,
        description: `${config.name} cache cluster`,
        engine: config.engine,
        node_type: config.nodeType,
        num_cache_clusters: config.clusterMode ? undefined : config.numNodes,
        automatic_failover_enabled: config.numNodes > 1,
        at_rest_encryption_enabled: true,
        transit_encryption_enabled: true,
        cluster_mode: config.clusterMode ? {
          num_node_groups: config.numNodes,
          replicas_per_node_group: 1,
        } : undefined,
      },
    });

    return this;
  }

  addS3Bucket(config: {
    name: string;
    versioning?: boolean;
    replication?: { destinationBucket: string; destinationRegion: string };
    lifecycle?: Array<{ prefix: string; transitionDays: number; storageClass: string }>;
  }): this {
    this.stack.resources.push({
      type: 'aws_s3_bucket',
      name: config.name,
      provider: 'aws',
      properties: {
        bucket: config.name,
        versioning: config.versioning ? { enabled: true } : undefined,
        replication_configuration: config.replication ? {
          role: `\${aws_iam_role.${config.name}-replication.arn}`,
          rules: [{
            destination: {
              bucket: config.replication.destinationBucket,
              storage_class: 'STANDARD',
            },
            status: 'Enabled',
          }],
        } : undefined,
        lifecycle_rule: config.lifecycle?.map(rule => ({
          prefix: rule.prefix,
          enabled: true,
          transition: {
            days: rule.transitionDays,
            storage_class: rule.storageClass,
          },
        })),
      },
    });

    return this;
  }

  addCloudFront(config: {
    name: string;
    origins: Array<{ id: string; domainName: string; type: 's3' | 'custom' }>;
    defaultCacheBehavior: { targetOriginId: string; viewerProtocolPolicy: string };
    priceClass?: string;
  }): this {
    this.stack.resources.push({
      type: 'aws_cloudfront_distribution',
      name: config.name,
      provider: 'aws',
      properties: {
        enabled: true,
        price_class: config.priceClass || 'PriceClass_All',
        origin: config.origins.map(o => ({
          origin_id: o.id,
          domain_name: o.domainName,
          ...(o.type === 's3' ? {
            s3_origin_config: { origin_access_identity: '' },
          } : {
            custom_origin_config: {
              http_port: 80,
              https_port: 443,
              origin_protocol_policy: 'https-only',
              origin_ssl_protocols: ['TLSv1.2'],
            },
          }),
        })),
        default_cache_behavior: {
          target_origin_id: config.defaultCacheBehavior.targetOriginId,
          viewer_protocol_policy: config.defaultCacheBehavior.viewerProtocolPolicy,
          allowed_methods: ['GET', 'HEAD', 'OPTIONS'],
          cached_methods: ['GET', 'HEAD'],
          compress: true,
          default_ttl: 86400,
        },
        viewer_certificate: {
          cloudfront_default_certificate: true,
        },
        restrictions: {
          geo_restriction: { restriction_type: 'none' },
        },
      },
    });

    return this;
  }

  output(name: string, value: string): this {
    this.stack.outputs[name] = value;
    return this;
  }

  build(): IaCStack {
    return this.stack;
  }
}

interface IaCProvider {
  create(resource: IaCResource): Promise<void>;
  update(resource: IaCResource): Promise<void>;
  delete(resource: IaCResource): Promise<void>;
}

class AWSProvider implements IaCProvider {
  async create(resource: IaCResource): Promise<void> {
    console.log(`[AWS] Creating ${resource.type}: ${resource.name}`);
  }
  async update(resource: IaCResource): Promise<void> {
    console.log(`[AWS] Updating ${resource.type}: ${resource.name}`);
  }
  async delete(resource: IaCResource): Promise<void> {
    console.log(`[AWS] Deleting ${resource.type}: ${resource.name}`);
  }
}

class GCPProvider implements IaCProvider {
  async create(resource: IaCResource): Promise<void> {
    console.log(`[GCP] Creating ${resource.type}: ${resource.name}`);
  }
  async update(resource: IaCResource): Promise<void> {
    console.log(`[GCP] Updating ${resource.type}: ${resource.name}`);
  }
  async delete(resource: IaCResource): Promise<void> {
    console.log(`[GCP] Deleting ${resource.type}: ${resource.name}`);
  }
}

class KubernetesProvider implements IaCProvider {
  async create(resource: IaCResource): Promise<void> {
    console.log(`[K8s] Creating ${resource.type}: ${resource.name}`);
  }
  async update(resource: IaCResource): Promise<void> {
    console.log(`[K8s] Updating ${resource.type}: ${resource.name}`);
  }
  async delete(resource: IaCResource): Promise<void> {
    console.log(`[K8s] Deleting ${resource.type}: ${resource.name}`);
  }
}

// ============================================================================
// CI/CD Pipeline
// ============================================================================

interface Pipeline {
  name: string;
  trigger: PipelineTrigger;
  stages: PipelineStage[];
  environment?: Record<string, string>;
  secrets?: string[];
  notifications?: NotificationConfig;
}

interface PipelineTrigger {
  type: 'push' | 'pull_request' | 'tag' | 'schedule' | 'manual';
  branches?: string[];
  paths?: string[];
  schedule?: string;
}

interface PipelineStage {
  name: string;
  jobs: PipelineJob[];
  dependsOn?: string[];
  condition?: string;
}

interface PipelineJob {
  name: string;
  runner: string;
  steps: PipelineStep[];
  services?: Record<string, ServiceConfig>;
  artifacts?: ArtifactConfig;
  cache?: CacheConfig;
  timeout?: number;
  retries?: number;
}

interface PipelineStep {
  name: string;
  run?: string;
  uses?: string;
  with?: Record<string, any>;
  env?: Record<string, string>;
  if?: string;
}

interface ServiceConfig {
  image: string;
  ports?: number[];
  env?: Record<string, string>;
}

interface ArtifactConfig {
  paths: string[];
  expireDays?: number;
}

interface CacheConfig {
  key: string;
  paths: string[];
}

interface NotificationConfig {
  slack?: { channel: string; onFailure: boolean; onSuccess: boolean };
  email?: { recipients: string[]; onFailure: boolean };
}

export class CICDPipelineGenerator {
  generateGitHubActions(pipeline: Pipeline): string {
    let yaml = `name: ${pipeline.name}\n\n`;

    // Triggers
    yaml += `on:\n`;
    if (pipeline.trigger.type === 'push') {
      yaml += `  push:\n`;
      if (pipeline.trigger.branches) {
        yaml += `    branches:\n`;
        pipeline.trigger.branches.forEach(b => yaml += `      - ${b}\n`);
      }
      if (pipeline.trigger.paths) {
        yaml += `    paths:\n`;
        pipeline.trigger.paths.forEach(p => yaml += `      - ${p}\n`);
      }
    } else if (pipeline.trigger.type === 'pull_request') {
      yaml += `  pull_request:\n`;
      if (pipeline.trigger.branches) {
        yaml += `    branches:\n`;
        pipeline.trigger.branches.forEach(b => yaml += `      - ${b}\n`);
      }
    } else if (pipeline.trigger.type === 'schedule') {
      yaml += `  schedule:\n`;
      yaml += `    - cron: '${pipeline.trigger.schedule}'\n`;
    }
    yaml += `  workflow_dispatch:\n\n`;

    // Environment
    if (pipeline.environment) {
      yaml += `env:\n`;
      for (const [key, value] of Object.entries(pipeline.environment)) {
        yaml += `  ${key}: ${value}\n`;
      }
      yaml += '\n';
    }

    // Jobs
    yaml += `jobs:\n`;

    for (const stage of pipeline.stages) {
      for (const job of stage.jobs) {
        yaml += `  ${job.name.replace(/\s+/g, '-').toLowerCase()}:\n`;
        yaml += `    name: ${job.name}\n`;
        yaml += `    runs-on: ${job.runner}\n`;

        if (stage.dependsOn?.length) {
          yaml += `    needs: [${stage.dependsOn.join(', ')}]\n`;
        }

        if (stage.condition) {
          yaml += `    if: ${stage.condition}\n`;
        }

        if (job.timeout) {
          yaml += `    timeout-minutes: ${job.timeout}\n`;
        }

        if (job.services) {
          yaml += `    services:\n`;
          for (const [name, config] of Object.entries(job.services)) {
            yaml += `      ${name}:\n`;
            yaml += `        image: ${config.image}\n`;
            if (config.ports) {
              yaml += `        ports:\n`;
              config.ports.forEach(p => yaml += `          - ${p}:${p}\n`);
            }
            if (config.env) {
              yaml += `        env:\n`;
              for (const [k, v] of Object.entries(config.env)) {
                yaml += `          ${k}: ${v}\n`;
              }
            }
          }
        }

        yaml += `    steps:\n`;
        for (const step of job.steps) {
          yaml += `      - name: ${step.name}\n`;
          if (step.uses) {
            yaml += `        uses: ${step.uses}\n`;
            if (step.with) {
              yaml += `        with:\n`;
              for (const [k, v] of Object.entries(step.with)) {
                yaml += `          ${k}: ${v}\n`;
              }
            }
          }
          if (step.run) {
            yaml += `        run: |\n`;
            step.run.split('\n').forEach(line => yaml += `          ${line}\n`);
          }
          if (step.env) {
            yaml += `        env:\n`;
            for (const [k, v] of Object.entries(step.env)) {
              yaml += `          ${k}: ${v}\n`;
            }
          }
          if (step.if) {
            yaml += `        if: ${step.if}\n`;
          }
        }

        if (job.artifacts) {
          yaml += `      - name: Upload artifacts\n`;
          yaml += `        uses: actions/upload-artifact@v3\n`;
          yaml += `        with:\n`;
          yaml += `          name: ${job.name.replace(/\s+/g, '-').toLowerCase()}-artifacts\n`;
          yaml += `          path: |\n`;
          job.artifacts.paths.forEach(p => yaml += `            ${p}\n`);
        }

        yaml += '\n';
      }
    }

    return yaml;
  }

  generateGitLabCI(pipeline: Pipeline): string {
    let yaml = `# ${pipeline.name}\n\n`;

    yaml += `stages:\n`;
    pipeline.stages.forEach(s => yaml += `  - ${s.name}\n`);
    yaml += '\n';

    if (pipeline.environment) {
      yaml += `variables:\n`;
      for (const [key, value] of Object.entries(pipeline.environment)) {
        yaml += `  ${key}: "${value}"\n`;
      }
      yaml += '\n';
    }

    for (const stage of pipeline.stages) {
      for (const job of stage.jobs) {
        yaml += `${job.name.replace(/\s+/g, '_').toLowerCase()}:\n`;
        yaml += `  stage: ${stage.name}\n`;
        yaml += `  image: ${job.runner}\n`;

        if (job.services) {
          yaml += `  services:\n`;
          for (const [, config] of Object.entries(job.services)) {
            yaml += `    - name: ${config.image}\n`;
          }
        }

        if (stage.dependsOn?.length) {
          yaml += `  needs:\n`;
          stage.dependsOn.forEach(d => yaml += `    - ${d}\n`);
        }

        if (job.cache) {
          yaml += `  cache:\n`;
          yaml += `    key: ${job.cache.key}\n`;
          yaml += `    paths:\n`;
          job.cache.paths.forEach(p => yaml += `      - ${p}\n`);
        }

        yaml += `  script:\n`;
        for (const step of job.steps) {
          if (step.run) {
            step.run.split('\n').forEach(line => {
              if (line.trim()) yaml += `    - ${line.trim()}\n`;
            });
          }
        }

        if (job.artifacts) {
          yaml += `  artifacts:\n`;
          yaml += `    paths:\n`;
          job.artifacts.paths.forEach(p => yaml += `      - ${p}\n`);
          if (job.artifacts.expireDays) {
            yaml += `    expire_in: ${job.artifacts.expireDays} days\n`;
          }
        }

        yaml += '\n';
      }
    }

    return yaml;
  }
}

// Pre-defined pipelines
export const cicdPipelines: Pipeline[] = [
  {
    name: 'Main CI/CD Pipeline',
    trigger: {
      type: 'push',
      branches: ['main', 'develop'],
      paths: ['src/**', 'package.json'],
    },
    environment: {
      NODE_VERSION: '20',
      DOCKER_REGISTRY: 'ghcr.io/mycelix',
    },
    stages: [
      {
        name: 'test',
        jobs: [
          {
            name: 'Unit Tests',
            runner: 'ubuntu-latest',
            services: {
              postgres: {
                image: 'postgres:15',
                ports: [5432],
                env: { POSTGRES_PASSWORD: 'test' },
              },
              redis: {
                image: 'redis:7',
                ports: [6379],
              },
            },
            cache: {
              key: 'npm-${{ hashFiles("package-lock.json") }}',
              paths: ['node_modules'],
            },
            steps: [
              { name: 'Checkout', uses: 'actions/checkout@v4' },
              { name: 'Setup Node', uses: 'actions/setup-node@v4', with: { 'node-version': '20' } },
              { name: 'Install dependencies', run: 'npm ci' },
              { name: 'Run tests', run: 'npm run test:coverage' },
              { name: 'Upload coverage', uses: 'codecov/codecov-action@v3' },
            ],
            artifacts: {
              paths: ['coverage/'],
              expireDays: 7,
            },
          },
          {
            name: 'Lint & Type Check',
            runner: 'ubuntu-latest',
            steps: [
              { name: 'Checkout', uses: 'actions/checkout@v4' },
              { name: 'Setup Node', uses: 'actions/setup-node@v4', with: { 'node-version': '20' } },
              { name: 'Install dependencies', run: 'npm ci' },
              { name: 'Lint', run: 'npm run lint' },
              { name: 'Type check', run: 'npm run typecheck' },
            ],
          },
          {
            name: 'Security Scan',
            runner: 'ubuntu-latest',
            steps: [
              { name: 'Checkout', uses: 'actions/checkout@v4' },
              { name: 'Run Snyk', uses: 'snyk/actions/node@master', env: { SNYK_TOKEN: '${{ secrets.SNYK_TOKEN }}' } },
              { name: 'Run Trivy', uses: 'aquasecurity/trivy-action@master', with: { 'scan-type': 'fs' } },
            ],
          },
        ],
      },
      {
        name: 'build',
        dependsOn: ['test'],
        jobs: [
          {
            name: 'Build Docker Image',
            runner: 'ubuntu-latest',
            steps: [
              { name: 'Checkout', uses: 'actions/checkout@v4' },
              { name: 'Set up Docker Buildx', uses: 'docker/setup-buildx-action@v3' },
              { name: 'Login to Registry', uses: 'docker/login-action@v3', with: { registry: 'ghcr.io', username: '${{ github.actor }}', password: '${{ secrets.GITHUB_TOKEN }}' } },
              { name: 'Build and push', uses: 'docker/build-push-action@v5', with: { push: true, tags: 'ghcr.io/mycelix/api:${{ github.sha }}', cache_from: 'type=gha', cache_to: 'type=gha,mode=max' } },
            ],
          },
        ],
      },
      {
        name: 'deploy-staging',
        dependsOn: ['build'],
        condition: "github.ref == 'refs/heads/develop'",
        jobs: [
          {
            name: 'Deploy to Staging',
            runner: 'ubuntu-latest',
            steps: [
              { name: 'Checkout', uses: 'actions/checkout@v4' },
              { name: 'Configure kubectl', uses: 'azure/k8s-set-context@v3', with: { kubeconfig: '${{ secrets.KUBE_CONFIG_STAGING }}' } },
              { name: 'Deploy', run: 'kubectl set image deployment/api api=ghcr.io/mycelix/api:${{ github.sha }} -n staging' },
              { name: 'Wait for rollout', run: 'kubectl rollout status deployment/api -n staging --timeout=5m' },
            ],
          },
        ],
      },
      {
        name: 'deploy-production',
        dependsOn: ['build'],
        condition: "github.ref == 'refs/heads/main'",
        jobs: [
          {
            name: 'Canary Deployment',
            runner: 'ubuntu-latest',
            steps: [
              { name: 'Checkout', uses: 'actions/checkout@v4' },
              { name: 'Configure kubectl', uses: 'azure/k8s-set-context@v3', with: { kubeconfig: '${{ secrets.KUBE_CONFIG_PROD }}' } },
              { name: 'Deploy canary (10%)', run: 'kubectl apply -f k8s/canary.yaml\nkubectl set image deployment/api-canary api=ghcr.io/mycelix/api:${{ github.sha }} -n production' },
              { name: 'Monitor canary', run: 'sleep 300 && ./scripts/check-canary-health.sh' },
              { name: 'Promote to production', run: 'kubectl set image deployment/api api=ghcr.io/mycelix/api:${{ github.sha }} -n production', if: 'success()' },
              { name: 'Rollback on failure', run: 'kubectl rollout undo deployment/api-canary -n production', if: 'failure()' },
            ],
          },
        ],
      },
    ],
    notifications: {
      slack: { channel: '#deployments', onFailure: true, onSuccess: true },
    },
  },
];

// ============================================================================
// Multi-Region Architecture
// ============================================================================

interface Region {
  name: string;
  provider: 'aws' | 'gcp' | 'azure';
  primary: boolean;
  endpoints: {
    api: string;
    cdn: string;
    database: string;
  };
  services: string[];
  healthCheck: string;
}

interface MultiRegionConfig {
  regions: Region[];
  routing: RoutingStrategy;
  failover: FailoverConfig;
  replication: ReplicationConfig;
}

interface RoutingStrategy {
  type: 'latency' | 'geolocation' | 'weighted' | 'failover';
  healthCheckInterval: number;
  weights?: Record<string, number>;
  geoMapping?: Record<string, string>;
}

interface FailoverConfig {
  automatic: boolean;
  threshold: number;
  cooldown: number;
  notifications: string[];
}

interface ReplicationConfig {
  database: {
    type: 'async' | 'sync';
    lagThreshold: number;
  };
  cache: {
    strategy: 'invalidate' | 'replicate';
  };
  storage: {
    type: 'cross-region' | 'multi-master';
  };
}

export class MultiRegionManager extends EventEmitter {
  private config: MultiRegionConfig;
  private regionHealth: Map<string, { healthy: boolean; latency: number; lastCheck: Date }> = new Map();

  constructor(config: MultiRegionConfig) {
    super();
    this.config = config;
    this.startHealthMonitoring();
  }

  private startHealthMonitoring(): void {
    setInterval(async () => {
      for (const region of this.config.regions) {
        await this.checkRegionHealth(region);
      }
      this.evaluateFailover();
    }, this.config.routing.healthCheckInterval * 1000);
  }

  private async checkRegionHealth(region: Region): Promise<void> {
    const startTime = Date.now();

    try {
      const response = await fetch(region.healthCheck, { method: 'GET' });
      const latency = Date.now() - startTime;

      this.regionHealth.set(region.name, {
        healthy: response.ok,
        latency,
        lastCheck: new Date(),
      });

      if (!response.ok) {
        this.emit('region_unhealthy', region.name);
      }
    } catch (error) {
      this.regionHealth.set(region.name, {
        healthy: false,
        latency: -1,
        lastCheck: new Date(),
      });
      this.emit('region_unreachable', region.name);
    }
  }

  private evaluateFailover(): void {
    if (!this.config.failover.automatic) return;

    const primaryRegion = this.config.regions.find(r => r.primary);
    if (!primaryRegion) return;

    const primaryHealth = this.regionHealth.get(primaryRegion.name);

    if (primaryHealth && !primaryHealth.healthy) {
      // Find best failover target
      let bestRegion: Region | null = null;
      let bestLatency = Infinity;

      for (const region of this.config.regions) {
        if (region.primary) continue;

        const health = this.regionHealth.get(region.name);
        if (health?.healthy && health.latency < bestLatency) {
          bestRegion = region;
          bestLatency = health.latency;
        }
      }

      if (bestRegion) {
        this.initiateFailover(primaryRegion, bestRegion);
      }
    }
  }

  private async initiateFailover(from: Region, to: Region): Promise<void> {
    console.log(`Initiating failover from ${from.name} to ${to.name}`);
    this.emit('failover_started', { from: from.name, to: to.name });

    // Update DNS routing
    await this.updateDNSRouting(to);

    // Promote database replica
    await this.promoteDatabase(to);

    // Update CDN origin
    await this.updateCDNOrigin(to);

    // Notify
    for (const notification of this.config.failover.notifications) {
      await this.sendNotification(notification, `Failover completed: ${from.name} -> ${to.name}`);
    }

    this.emit('failover_completed', { from: from.name, to: to.name });
  }

  private async updateDNSRouting(targetRegion: Region): Promise<void> {
    console.log(`Updating DNS routing to ${targetRegion.name}`);
    // Would update Route53, Cloud DNS, etc.
  }

  private async promoteDatabase(targetRegion: Region): Promise<void> {
    console.log(`Promoting database replica in ${targetRegion.name}`);
    // Would promote RDS replica, Cloud SQL replica, etc.
  }

  private async updateCDNOrigin(targetRegion: Region): Promise<void> {
    console.log(`Updating CDN origin to ${targetRegion.name}`);
    // Would update CloudFront, Cloud CDN, etc.
  }

  private async sendNotification(target: string, message: string): Promise<void> {
    console.log(`Notification to ${target}: ${message}`);
  }

  routeRequest(clientLocation: { lat: number; lng: number }): Region {
    switch (this.config.routing.type) {
      case 'latency':
        return this.routeByLatency();
      case 'geolocation':
        return this.routeByGeolocation(clientLocation);
      case 'weighted':
        return this.routeByWeight();
      case 'failover':
        return this.routeByFailover();
      default:
        return this.config.regions.find(r => r.primary)!;
    }
  }

  private routeByLatency(): Region {
    let bestRegion = this.config.regions[0];
    let bestLatency = Infinity;

    for (const region of this.config.regions) {
      const health = this.regionHealth.get(region.name);
      if (health?.healthy && health.latency < bestLatency) {
        bestRegion = region;
        bestLatency = health.latency;
      }
    }

    return bestRegion;
  }

  private routeByGeolocation(location: { lat: number; lng: number }): Region {
    // Simple continent-based routing
    const geoMapping = this.config.routing.geoMapping || {};

    // Would use actual geolocation logic
    const continent = this.determineContinent(location);
    const regionName = geoMapping[continent];

    return this.config.regions.find(r => r.name === regionName) || this.config.regions[0];
  }

  private determineContinent(location: { lat: number; lng: number }): string {
    // Simplified continent detection
    if (location.lng < -30) return 'americas';
    if (location.lng > 60) return 'asia';
    return 'europe';
  }

  private routeByWeight(): Region {
    const weights = this.config.routing.weights || {};
    const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
    let random = Math.random() * totalWeight;

    for (const region of this.config.regions) {
      const weight = weights[region.name] || 0;
      random -= weight;
      if (random <= 0) return region;
    }

    return this.config.regions[0];
  }

  private routeByFailover(): Region {
    for (const region of this.config.regions) {
      const health = this.regionHealth.get(region.name);
      if (health?.healthy) return region;
    }
    return this.config.regions[0];
  }

  getRegionStatus(): object {
    return {
      regions: this.config.regions.map(r => ({
        name: r.name,
        primary: r.primary,
        ...this.regionHealth.get(r.name),
      })),
      routing: this.config.routing.type,
    };
  }
}

// Multi-region configuration
export const multiRegionConfig: MultiRegionConfig = {
  regions: [
    {
      name: 'us-east-1',
      provider: 'aws',
      primary: true,
      endpoints: {
        api: 'https://api-us-east.mycelix.io',
        cdn: 'https://cdn-us-east.mycelix.io',
        database: 'postgres-us-east.mycelix.internal',
      },
      services: ['api', 'streaming', 'search', 'recommendations'],
      healthCheck: 'https://api-us-east.mycelix.io/health',
    },
    {
      name: 'eu-west-1',
      provider: 'aws',
      primary: false,
      endpoints: {
        api: 'https://api-eu-west.mycelix.io',
        cdn: 'https://cdn-eu-west.mycelix.io',
        database: 'postgres-eu-west.mycelix.internal',
      },
      services: ['api', 'streaming', 'search', 'recommendations'],
      healthCheck: 'https://api-eu-west.mycelix.io/health',
    },
    {
      name: 'ap-southeast-1',
      provider: 'aws',
      primary: false,
      endpoints: {
        api: 'https://api-ap-southeast.mycelix.io',
        cdn: 'https://cdn-ap-southeast.mycelix.io',
        database: 'postgres-ap-southeast.mycelix.internal',
      },
      services: ['api', 'streaming', 'search'],
      healthCheck: 'https://api-ap-southeast.mycelix.io/health',
    },
  ],
  routing: {
    type: 'latency',
    healthCheckInterval: 30,
  },
  failover: {
    automatic: true,
    threshold: 3,
    cooldown: 300,
    notifications: ['ops@mycelix.io', 'slack:#incidents'],
  },
  replication: {
    database: { type: 'async', lagThreshold: 1000 },
    cache: { strategy: 'invalidate' },
    storage: { type: 'cross-region' },
  },
};

// ============================================================================
// Feature Flags
// ============================================================================

interface FeatureFlag {
  key: string;
  name: string;
  description: string;
  type: 'boolean' | 'string' | 'number' | 'json';
  defaultValue: any;
  enabled: boolean;
  targeting: TargetingRule[];
  rollout?: RolloutConfig;
  variants?: FlagVariant[];
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
}

interface TargetingRule {
  id: string;
  name: string;
  conditions: TargetingCondition[];
  variation: string | number;
  percentage?: number;
}

interface TargetingCondition {
  attribute: string;
  operator: 'equals' | 'notEquals' | 'contains' | 'startsWith' | 'in' | 'notIn' | 'greaterThan' | 'lessThan';
  value: any;
}

interface RolloutConfig {
  type: 'percentage' | 'schedule' | 'gradual';
  percentage?: number;
  schedule?: { start: Date; end: Date };
  gradual?: { startPercentage: number; endPercentage: number; durationHours: number };
}

interface FlagVariant {
  key: string;
  name: string;
  value: any;
  weight: number;
}

interface EvaluationContext {
  userId?: string;
  userEmail?: string;
  userCountry?: string;
  userPlan?: string;
  deviceType?: string;
  appVersion?: string;
  custom?: Record<string, any>;
}

interface FlagEvaluation {
  flagKey: string;
  value: any;
  variation: string;
  reason: string;
}

export class FeatureFlagService extends EventEmitter {
  private flags: Map<string, FeatureFlag> = new Map();
  private evaluationCache: Map<string, { value: any; expiry: number }> = new Map();
  private analytics: FlagAnalytics;

  constructor() {
    super();
    this.analytics = new FlagAnalytics();
  }

  createFlag(flag: Omit<FeatureFlag, 'createdAt' | 'updatedAt'>): FeatureFlag {
    const fullFlag: FeatureFlag = {
      ...flag,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    this.flags.set(flag.key, fullFlag);
    this.emit('flag_created', flag.key);
    return fullFlag;
  }

  updateFlag(key: string, updates: Partial<FeatureFlag>): FeatureFlag | null {
    const flag = this.flags.get(key);
    if (!flag) return null;

    const updated = { ...flag, ...updates, updatedAt: new Date() };
    this.flags.set(key, updated);
    this.invalidateCache(key);
    this.emit('flag_updated', key);
    return updated;
  }

  deleteFlag(key: string): boolean {
    const deleted = this.flags.delete(key);
    if (deleted) {
      this.invalidateCache(key);
      this.emit('flag_deleted', key);
    }
    return deleted;
  }

  evaluate(key: string, context: EvaluationContext, defaultValue?: any): FlagEvaluation {
    const flag = this.flags.get(key);

    if (!flag) {
      return {
        flagKey: key,
        value: defaultValue,
        variation: 'default',
        reason: 'FLAG_NOT_FOUND',
      };
    }

    if (!flag.enabled) {
      return {
        flagKey: key,
        value: flag.defaultValue,
        variation: 'off',
        reason: 'FLAG_DISABLED',
      };
    }

    // Check cache
    const cacheKey = this.getCacheKey(key, context);
    const cached = this.evaluationCache.get(cacheKey);
    if (cached && cached.expiry > Date.now()) {
      return {
        flagKey: key,
        value: cached.value,
        variation: 'cached',
        reason: 'CACHE_HIT',
      };
    }

    // Evaluate targeting rules
    for (const rule of flag.targeting) {
      if (this.evaluateRule(rule, context)) {
        const value = this.getVariationValue(flag, rule.variation);
        this.cacheEvaluation(cacheKey, value);
        this.analytics.recordEvaluation(key, rule.id, context);
        return {
          flagKey: key,
          value,
          variation: String(rule.variation),
          reason: `RULE_MATCH:${rule.name}`,
        };
      }
    }

    // Check rollout
    if (flag.rollout) {
      const inRollout = this.evaluateRollout(flag.rollout, context);
      if (inRollout) {
        const value = flag.variants ? this.selectVariant(flag.variants, context) : true;
        this.cacheEvaluation(cacheKey, value);
        return {
          flagKey: key,
          value,
          variation: 'rollout',
          reason: 'ROLLOUT_INCLUDED',
        };
      }
    }

    // Default
    this.cacheEvaluation(cacheKey, flag.defaultValue);
    return {
      flagKey: key,
      value: flag.defaultValue,
      variation: 'default',
      reason: 'DEFAULT_VALUE',
    };
  }

  evaluateAll(context: EvaluationContext): Record<string, any> {
    const results: Record<string, any> = {};
    for (const [key] of this.flags) {
      results[key] = this.evaluate(key, context).value;
    }
    return results;
  }

  private evaluateRule(rule: TargetingRule, context: EvaluationContext): boolean {
    // Check percentage if specified
    if (rule.percentage !== undefined) {
      const hash = this.hashContext(context);
      if (hash > rule.percentage) return false;
    }

    // Evaluate all conditions (AND logic)
    return rule.conditions.every(condition => this.evaluateCondition(condition, context));
  }

  private evaluateCondition(condition: TargetingCondition, context: EvaluationContext): boolean {
    const contextValue = this.getContextValue(context, condition.attribute);

    switch (condition.operator) {
      case 'equals':
        return contextValue === condition.value;
      case 'notEquals':
        return contextValue !== condition.value;
      case 'contains':
        return String(contextValue).includes(condition.value);
      case 'startsWith':
        return String(contextValue).startsWith(condition.value);
      case 'in':
        return Array.isArray(condition.value) && condition.value.includes(contextValue);
      case 'notIn':
        return Array.isArray(condition.value) && !condition.value.includes(contextValue);
      case 'greaterThan':
        return Number(contextValue) > Number(condition.value);
      case 'lessThan':
        return Number(contextValue) < Number(condition.value);
      default:
        return false;
    }
  }

  private evaluateRollout(rollout: RolloutConfig, context: EvaluationContext): boolean {
    switch (rollout.type) {
      case 'percentage':
        const hash = this.hashContext(context);
        return hash <= (rollout.percentage || 0);

      case 'schedule':
        const now = new Date();
        return rollout.schedule!.start <= now && now <= rollout.schedule!.end;

      case 'gradual':
        const elapsed = (Date.now() - rollout.gradual!.durationHours * 3600000) / (rollout.gradual!.durationHours * 3600000);
        const currentPercentage = rollout.gradual!.startPercentage + (rollout.gradual!.endPercentage - rollout.gradual!.startPercentage) * Math.min(elapsed, 1);
        return this.hashContext(context) <= currentPercentage;

      default:
        return false;
    }
  }

  private selectVariant(variants: FlagVariant[], context: EvaluationContext): any {
    const hash = this.hashContext(context);
    const totalWeight = variants.reduce((sum, v) => sum + v.weight, 0);
    let threshold = 0;

    for (const variant of variants) {
      threshold += (variant.weight / totalWeight) * 100;
      if (hash <= threshold) {
        return variant.value;
      }
    }

    return variants[0].value;
  }

  private getVariationValue(flag: FeatureFlag, variation: string | number): any {
    if (flag.variants) {
      const variant = flag.variants.find(v => v.key === variation);
      if (variant) return variant.value;
    }
    return variation;
  }

  private getContextValue(context: EvaluationContext, attribute: string): any {
    if (attribute.startsWith('custom.')) {
      return context.custom?.[attribute.slice(7)];
    }
    return (context as any)[attribute];
  }

  private hashContext(context: EvaluationContext): number {
    // Simple hash for percentage bucketing
    const str = context.userId || context.userEmail || JSON.stringify(context);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash) % 100;
  }

  private getCacheKey(flagKey: string, context: EvaluationContext): string {
    return `${flagKey}:${context.userId || 'anonymous'}`;
  }

  private cacheEvaluation(key: string, value: any): void {
    this.evaluationCache.set(key, {
      value,
      expiry: Date.now() + 60000, // 1 minute cache
    });
  }

  private invalidateCache(flagKey: string): void {
    for (const [key] of this.evaluationCache) {
      if (key.startsWith(`${flagKey}:`)) {
        this.evaluationCache.delete(key);
      }
    }
  }

  getFlags(): FeatureFlag[] {
    return Array.from(this.flags.values());
  }

  getFlag(key: string): FeatureFlag | undefined {
    return this.flags.get(key);
  }

  getAnalytics(flagKey: string): object {
    return this.analytics.getStats(flagKey);
  }
}

class FlagAnalytics {
  private evaluations: Map<string, Array<{ ruleId: string; timestamp: Date; context: EvaluationContext }>> = new Map();

  recordEvaluation(flagKey: string, ruleId: string, context: EvaluationContext): void {
    if (!this.evaluations.has(flagKey)) {
      this.evaluations.set(flagKey, []);
    }
    this.evaluations.get(flagKey)!.push({ ruleId, timestamp: new Date(), context });
  }

  getStats(flagKey: string): object {
    const evals = this.evaluations.get(flagKey) || [];
    const ruleHits: Record<string, number> = {};

    for (const eval_ of evals) {
      ruleHits[eval_.ruleId] = (ruleHits[eval_.ruleId] || 0) + 1;
    }

    return {
      totalEvaluations: evals.length,
      ruleHits,
      last24Hours: evals.filter(e => e.timestamp > new Date(Date.now() - 86400000)).length,
    };
  }
}

// Pre-defined feature flags
export const defaultFeatureFlags: Omit<FeatureFlag, 'createdAt' | 'updatedAt'>[] = [
  {
    key: 'new_player_ui',
    name: 'New Player UI',
    description: 'Enable the redesigned audio player interface',
    type: 'boolean',
    defaultValue: false,
    enabled: true,
    targeting: [
      {
        id: 'beta_users',
        name: 'Beta Users',
        conditions: [{ attribute: 'custom.betaUser', operator: 'equals', value: true }],
        variation: 'true',
      },
    ],
    rollout: { type: 'gradual', gradual: { startPercentage: 5, endPercentage: 100, durationHours: 168 } },
    tags: ['ui', 'player'],
  },
  {
    key: 'ai_recommendations',
    name: 'AI Recommendations',
    description: 'Use ML-powered recommendation engine',
    type: 'boolean',
    defaultValue: false,
    enabled: true,
    targeting: [
      {
        id: 'premium_users',
        name: 'Premium Users',
        conditions: [{ attribute: 'userPlan', operator: 'in', value: ['premium', 'family', 'student'] }],
        variation: 'true',
      },
    ],
    tags: ['ml', 'recommendations'],
  },
  {
    key: 'spatial_audio',
    name: 'Spatial Audio',
    description: 'Enable spatial audio for supported tracks',
    type: 'boolean',
    defaultValue: false,
    enabled: true,
    targeting: [
      {
        id: 'supported_devices',
        name: 'Supported Devices',
        conditions: [{ attribute: 'deviceType', operator: 'in', value: ['ios', 'airpods'] }],
        variation: 'true',
      },
    ],
    tags: ['audio', 'premium'],
  },
  {
    key: 'streaming_quality',
    name: 'Streaming Quality',
    description: 'Default streaming quality setting',
    type: 'string',
    defaultValue: 'high',
    enabled: true,
    targeting: [],
    variants: [
      { key: 'low', name: 'Low (96kbps)', value: 'low', weight: 10 },
      { key: 'medium', name: 'Medium (160kbps)', value: 'medium', weight: 30 },
      { key: 'high', name: 'High (320kbps)', value: 'high', weight: 50 },
      { key: 'lossless', name: 'Lossless', value: 'lossless', weight: 10 },
    ],
    tags: ['streaming', 'quality'],
  },
];

// ============================================================================
// GitOps Configuration
// ============================================================================

interface GitOpsConfig {
  repository: string;
  branch: string;
  paths: {
    applications: string;
    infrastructure: string;
    config: string;
  };
  syncPolicy: {
    automated: boolean;
    selfHeal: boolean;
    prune: boolean;
    syncInterval: number;
  };
  notifications: {
    slack?: string;
    email?: string[];
  };
}

interface Application {
  name: string;
  namespace: string;
  source: {
    repoURL: string;
    path: string;
    targetRevision: string;
  };
  destination: {
    server: string;
    namespace: string;
  };
  syncPolicy?: {
    automated?: { prune: boolean; selfHeal: boolean };
    syncOptions?: string[];
  };
  healthChecks?: HealthCheck[];
}

interface HealthCheck {
  type: 'http' | 'tcp' | 'exec';
  endpoint?: string;
  port?: number;
  command?: string[];
  interval: number;
  timeout: number;
}

export class GitOpsManager {
  private config: GitOpsConfig;
  private applications: Map<string, Application> = new Map();

  constructor(config: GitOpsConfig) {
    this.config = config;
  }

  registerApplication(app: Application): void {
    this.applications.set(app.name, app);
  }

  generateArgoApplication(app: Application): string {
    return `apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ${app.name}
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: ${app.source.repoURL}
    path: ${app.source.path}
    targetRevision: ${app.source.targetRevision}
  destination:
    server: ${app.destination.server}
    namespace: ${app.destination.namespace}
  syncPolicy:
${app.syncPolicy?.automated ? `    automated:
      prune: ${app.syncPolicy.automated.prune}
      selfHeal: ${app.syncPolicy.automated.selfHeal}` : '    {}'}
${app.syncPolicy?.syncOptions ? `    syncOptions:
${app.syncPolicy.syncOptions.map(o => `      - ${o}`).join('\n')}` : ''}
`;
  }

  generateFluxKustomization(app: Application): string {
    return `apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: ${app.name}
  namespace: flux-system
spec:
  interval: ${this.config.syncPolicy.syncInterval}s
  path: ${app.source.path}
  prune: ${this.config.syncPolicy.prune}
  sourceRef:
    kind: GitRepository
    name: ${this.config.repository.split('/').pop()?.replace('.git', '')}
  targetNamespace: ${app.destination.namespace}
  healthChecks:
${app.healthChecks?.map(hc => `    - apiVersion: apps/v1
      kind: Deployment
      name: ${app.name}
      namespace: ${app.destination.namespace}`).join('\n') || '    []'}
`;
  }

  generateKubernetesManifests(app: Application): object {
    return {
      deployment: this.generateDeployment(app),
      service: this.generateService(app),
      ingress: this.generateIngress(app),
      hpa: this.generateHPA(app),
      pdb: this.generatePDB(app),
    };
  }

  private generateDeployment(app: Application): object {
    return {
      apiVersion: 'apps/v1',
      kind: 'Deployment',
      metadata: {
        name: app.name,
        namespace: app.namespace,
        labels: { app: app.name },
      },
      spec: {
        replicas: 3,
        selector: { matchLabels: { app: app.name } },
        template: {
          metadata: { labels: { app: app.name } },
          spec: {
            containers: [{
              name: app.name,
              image: `${app.source.repoURL}:${app.source.targetRevision}`,
              ports: [{ containerPort: 8080 }],
              resources: {
                requests: { cpu: '100m', memory: '256Mi' },
                limits: { cpu: '1000m', memory: '1Gi' },
              },
              livenessProbe: {
                httpGet: { path: '/health/live', port: 8080 },
                initialDelaySeconds: 30,
                periodSeconds: 10,
              },
              readinessProbe: {
                httpGet: { path: '/health/ready', port: 8080 },
                initialDelaySeconds: 5,
                periodSeconds: 5,
              },
            }],
          },
        },
      },
    };
  }

  private generateService(app: Application): object {
    return {
      apiVersion: 'v1',
      kind: 'Service',
      metadata: {
        name: app.name,
        namespace: app.namespace,
      },
      spec: {
        selector: { app: app.name },
        ports: [{ port: 80, targetPort: 8080 }],
        type: 'ClusterIP',
      },
    };
  }

  private generateIngress(app: Application): object {
    return {
      apiVersion: 'networking.k8s.io/v1',
      kind: 'Ingress',
      metadata: {
        name: app.name,
        namespace: app.namespace,
        annotations: {
          'kubernetes.io/ingress.class': 'nginx',
          'cert-manager.io/cluster-issuer': 'letsencrypt-prod',
        },
      },
      spec: {
        tls: [{ hosts: [`${app.name}.mycelix.io`], secretName: `${app.name}-tls` }],
        rules: [{
          host: `${app.name}.mycelix.io`,
          http: {
            paths: [{
              path: '/',
              pathType: 'Prefix',
              backend: { service: { name: app.name, port: { number: 80 } } },
            }],
          },
        }],
      },
    };
  }

  private generateHPA(app: Application): object {
    return {
      apiVersion: 'autoscaling/v2',
      kind: 'HorizontalPodAutoscaler',
      metadata: {
        name: app.name,
        namespace: app.namespace,
      },
      spec: {
        scaleTargetRef: {
          apiVersion: 'apps/v1',
          kind: 'Deployment',
          name: app.name,
        },
        minReplicas: 3,
        maxReplicas: 20,
        metrics: [
          { type: 'Resource', resource: { name: 'cpu', target: { type: 'Utilization', averageUtilization: 70 } } },
          { type: 'Resource', resource: { name: 'memory', target: { type: 'Utilization', averageUtilization: 80 } } },
        ],
      },
    };
  }

  private generatePDB(app: Application): object {
    return {
      apiVersion: 'policy/v1',
      kind: 'PodDisruptionBudget',
      metadata: {
        name: app.name,
        namespace: app.namespace,
      },
      spec: {
        minAvailable: '50%',
        selector: { matchLabels: { app: app.name } },
      },
    };
  }
}

// ============================================================================
// Export
// ============================================================================

export const createDeploymentInfrastructure = (): {
  iac: InfrastructureAsCode;
  cicd: CICDPipelineGenerator;
  multiRegion: MultiRegionManager;
  featureFlags: FeatureFlagService;
  gitOps: GitOpsManager;
} => {
  const iac = new InfrastructureAsCode();
  const cicd = new CICDPipelineGenerator();
  const multiRegion = new MultiRegionManager(multiRegionConfig);
  const featureFlags = new FeatureFlagService();
  const gitOps = new GitOpsManager({
    repository: 'https://github.com/mycelix/infrastructure',
    branch: 'main',
    paths: {
      applications: 'apps/',
      infrastructure: 'infra/',
      config: 'config/',
    },
    syncPolicy: {
      automated: true,
      selfHeal: true,
      prune: true,
      syncInterval: 300,
    },
    notifications: {
      slack: '#deployments',
    },
  });

  // Register default feature flags
  defaultFeatureFlags.forEach(flag => featureFlags.createFlag(flag));

  return { iac, cicd, multiRegion, featureFlags, gitOps };
};
