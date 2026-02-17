/**
 * Projects Zome Client
 *
 * Handles energy project registration and management.
 *
 * @module @mycelix/sdk/integrations/energy/zomes/projects
 */

import { EnergySdkError } from '../types';

import type {
  EnergyProject,
  RegisterProjectInput,
  UpdateProjectInput,
  ProjectSearchParams,
  ProjectStatistics,
  EnergySource,
  ProjectStatus,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Projects client
 */
export interface ProjectsClientConfig {
  roleId: string;
  zomeName: string;
}

const DEFAULT_CONFIG: ProjectsClientConfig = {
  roleId: 'energy',
  zomeName: 'projects',
};

/**
 * Client for energy project operations
 *
 * @example
 * ```typescript
 * const projects = new ProjectsClient(holochainClient);
 *
 * // Register a new solar project
 * const project = await projects.registerProject({
 *   name: 'Community Solar Farm',
 *   description: 'A 500kW community-owned solar installation',
 *   source: 'Solar',
 *   capacity_kw: 500,
 *   location: { lat: 32.95, lng: -96.73, address: 'Richardson, TX' },
 *   investment_goal: 750000,
 * });
 *
 * // Search for operational solar projects
 * const solarProjects = await projects.searchProjects({
 *   source: 'Solar',
 *   status: 'Operational',
 *   min_capacity_kw: 100,
 * });
 * ```
 */
export class ProjectsClient {
  private readonly config: ProjectsClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<ProjectsClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      throw new EnergySdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Extract entry from a Holochain record
   */
  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new EnergySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Project Operations
  // ============================================================================

  /**
   * Register a new energy project
   *
   * @param input - Project registration parameters
   * @returns The created project
   */
  async registerProject(input: RegisterProjectInput): Promise<EnergyProject> {
    const record = await this.call<HolochainRecord>('register_project', {
      ...input,
      investment_raised: 0,
      created_at: Date.now() * 1000,
      updated_at: Date.now() * 1000,
    });
    return this.extractEntry<EnergyProject>(record);
  }

  /**
   * Get a project by ID
   *
   * @param projectId - Project identifier
   * @returns The project or null if not found
   */
  async getProject(projectId: string): Promise<EnergyProject | null> {
    const record = await this.call<HolochainRecord | null>('get_project', projectId);
    if (!record) return null;
    return this.extractEntry<EnergyProject>(record);
  }

  /**
   * Update a project
   *
   * @param input - Update parameters
   * @returns The updated project
   */
  async updateProject(input: UpdateProjectInput): Promise<EnergyProject> {
    const record = await this.call<HolochainRecord>('update_project', {
      ...input,
      updated_at: Date.now() * 1000,
    });
    return this.extractEntry<EnergyProject>(record);
  }

  /**
   * Get projects by energy source
   *
   * @param source - Energy source type
   * @returns Array of projects
   */
  async getProjectsBySource(source: EnergySource): Promise<EnergyProject[]> {
    const records = await this.call<HolochainRecord[]>('get_projects_by_source', source);
    return records.map(r => this.extractEntry<EnergyProject>(r));
  }

  /**
   * Get projects by status
   *
   * @param status - Project status
   * @returns Array of projects
   */
  async getProjectsByStatus(status: ProjectStatus): Promise<EnergyProject[]> {
    const records = await this.call<HolochainRecord[]>('get_projects_by_status', status);
    return records.map(r => this.extractEntry<EnergyProject>(r));
  }

  /**
   * Get projects by DAO
   *
   * @param daoId - DAO identifier
   * @returns Array of projects owned by the DAO
   */
  async getProjectsByDao(daoId: string): Promise<EnergyProject[]> {
    const records = await this.call<HolochainRecord[]>('get_projects_by_dao', daoId);
    return records.map(r => this.extractEntry<EnergyProject>(r));
  }

  /**
   * Search projects with filters
   *
   * @param params - Search parameters
   * @returns Array of matching projects
   */
  async searchProjects(params: ProjectSearchParams): Promise<EnergyProject[]> {
    const records = await this.call<HolochainRecord[]>('search_projects', params);
    return records.map(r => this.extractEntry<EnergyProject>(r));
  }

  /**
   * Get all operational projects
   *
   * @returns Array of operational projects
   */
  async getOperationalProjects(): Promise<EnergyProject[]> {
    return this.getProjectsByStatus('Operational');
  }

  /**
   * Get projects seeking investment
   *
   * @returns Array of projects with open investment
   */
  async getInvestmentOpportunities(): Promise<EnergyProject[]> {
    const records = await this.call<HolochainRecord[]>('get_investment_opportunities', null);
    return records.map(r => this.extractEntry<EnergyProject>(r));
  }

  /**
   * Get project statistics
   *
   * @param projectId - Project identifier
   * @returns Project statistics
   */
  async getProjectStatistics(projectId: string): Promise<ProjectStatistics> {
    return this.call<ProjectStatistics>('get_project_statistics', projectId);
  }

  /**
   * Get projects near a location
   *
   * @param lat - Latitude
   * @param lng - Longitude
   * @param radiusKm - Search radius in kilometers
   * @returns Array of nearby projects
   */
  async getProjectsNearLocation(
    lat: number,
    lng: number,
    radiusKm: number = 50
  ): Promise<EnergyProject[]> {
    const records = await this.call<HolochainRecord[]>('get_projects_near_location', {
      lat,
      lng,
      radius_km: radiusKm,
    });
    return records.map(r => this.extractEntry<EnergyProject>(r));
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick project registration with auto-generated ID
   *
   * @param name - Project name
   * @param source - Energy source
   * @param capacityKw - Capacity in kW
   * @param location - Project location
   * @param description - Optional description
   * @returns The created project
   */
  async quickRegister(
    name: string,
    source: EnergySource,
    capacityKw: number,
    location: { lat: number; lng: number; address?: string },
    description?: string
  ): Promise<EnergyProject> {
    return this.registerProject({
      name,
      description: description || `${source} energy project - ${capacityKw}kW`,
      source,
      capacity_kw: capacityKw,
      location,
    });
  }

  /**
   * Calculate project capacity factor
   *
   * @param project - The project
   * @returns Capacity factor (0-1)
   */
  calculateCapacityFactor(project: EnergyProject): number {
    if (!project.annual_generation_mwh || project.capacity_kw <= 0) {
      return 0;
    }
    const maxAnnualMwh = (project.capacity_kw / 1000) * 8760;
    return project.annual_generation_mwh / maxAnnualMwh;
  }

  /**
   * Estimate annual generation for a project
   *
   * @param project - The project
   * @returns Estimated annual generation in MWh
   */
  estimateAnnualGeneration(project: EnergyProject): number {
    const capacityFactors: Record<EnergySource, number> = {
      Solar: 0.18,
      Wind: 0.35,
      Hydro: 0.45,
      Nuclear: 0.92,
      Geothermal: 0.90,
      Storage: 0.10,
      Biomass: 0.80,
      Grid: 1.0,
      Other: 0.25,
    };

    const cf = capacityFactors[project.source] || 0.25;
    return (project.capacity_kw / 1000) * 8760 * cf;
  }

  /**
   * Get investment progress percentage
   *
   * @param project - The project
   * @returns Investment progress (0-100)
   */
  getInvestmentProgress(project: EnergyProject): number {
    if (!project.investment_goal || project.investment_goal <= 0) {
      return 100;
    }
    return Math.min(100, (project.investment_raised / project.investment_goal) * 100);
  }

  /**
   * Check if project is accepting investments
   *
   * @param project - The project
   * @returns True if accepting investments
   */
  isAcceptingInvestments(project: EnergyProject): boolean {
    if (!project.investment_goal) return false;
    return (
      project.investment_raised < project.investment_goal &&
      (project.status === 'Proposed' || project.status === 'Planning' || project.status === 'Development')
    );
  }

  /**
   * Get status description
   *
   * @param status - Project status
   * @returns Human-readable description
   */
  getStatusDescription(status: ProjectStatus): string {
    const descriptions: Record<ProjectStatus, string> = {
      Proposed: 'Project has been proposed and is under review',
      Planning: 'Project is in planning and permitting phase',
      Development: 'Project is under development',
      Construction: 'Project is under construction',
      Operational: 'Project is operational and generating energy',
      Decommissioned: 'Project has been decommissioned',
    };
    return descriptions[status];
  }
}
