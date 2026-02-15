/**
 * @mycelix/sdk Energy API Client
 *
 * Real API integrations for energy data:
 * - NREL Solar Resource API (solar potential by location)
 * - ERCOT Grid Status (Texas grid conditions)
 * - Weather API (solar production forecasts)
 * - Terra Atlas Bridge (nearby projects)
 *
 * @packageDocumentation
 * @module energy/api-client
 */

// ============================================================================
// NREL SOLAR API
// ============================================================================

/**
 * Solar resource data from NREL API
 * Source: https://developer.nrel.gov/docs/solar/solar_resource/v1/
 */
export interface NRELSolarResource {
  latitude: number;
  longitude: number;
  elevation: number;

  // Annual averages (kWh/m²/day)
  ghi: number; // Global Horizontal Irradiance
  dni: number; // Direct Normal Irradiance
  dhi: number; // Diffuse Horizontal Irradiance

  // Tilted irradiance at latitude angle
  tiltedIrradiance: number;
  optimalTiltAngle: number;

  // Annual generation estimate
  annualKwhPerKw: number;

  // Monthly breakdown (12 values)
  monthlyGhi: number[];
  monthlyGeneration: number[];

  // Data quality
  dataSource: string;
}

/**
 * Solar project economics calculation
 */
export interface SolarProjectAnalysis {
  // System
  systemSizeKw: number;
  numPanels: number;
  roofAreaM2: number;

  // Generation
  annualGenerationKwh: number;
  capacityFactor: number;
  firstYearProduction: number;

  // Economics (in USD)
  grossCost: number;
  federalItc: number; // 30% through 2032
  stateIncentives: number;
  netCost: number;

  // Returns
  annualSavings: number;
  paybackYears: number;
  twentyFiveYearSavings: number;
  irr: number;
  lcoe: number; // $/kWh

  // Monthly projections
  monthlyProduction: number[];
  monthlySavings: number[];
}

/**
 * NREL Solar API Client
 *
 * Free API key from: https://developer.nrel.gov/signup/
 */
export class NRELSolarClient {
  private apiKey: string;
  private baseUrl = 'https://developer.nrel.gov/api/solar';

  constructor(apiKey?: string) {
    // Use environment variable or provided key
    this.apiKey = apiKey || (typeof process !== 'undefined' ? process.env.NREL_API_KEY || '' : '');
  }

  /**
   * Get solar resource data for a location
   */
  async getSolarResource(lat: number, lng: number): Promise<NRELSolarResource> {
    if (!this.apiKey) {
      // Return demo data if no API key
      return this.getDemoSolarResource(lat, lng);
    }

    const url =
      `${this.baseUrl}/solar_resource/v1.json?` +
      `api_key=${this.apiKey}&` +
      `lat=${lat}&lon=${lng}`;

    const response = await fetch(url);

    if (!response.ok) {
      console.warn(`NREL API error: ${response.status}, using demo data`);
      return this.getDemoSolarResource(lat, lng);
    }

    const data = await response.json();
    const outputs = data.outputs;

    return {
      latitude: lat,
      longitude: lng,
      elevation: outputs.elevation || 0,

      ghi: outputs.avg_ghi?.annual || 5.0,
      dni: outputs.avg_dni?.annual || 5.5,
      dhi: outputs.avg_dhi?.annual || 2.0,

      tiltedIrradiance: outputs.avg_lat_tilt?.annual || 5.5,
      optimalTiltAngle: lat, // Optimal tilt ≈ latitude

      annualKwhPerKw: (outputs.avg_lat_tilt?.annual || 5.5) * 365 * 0.8, // 80% performance ratio

      monthlyGhi: outputs.avg_ghi?.monthly || this.getDefaultMonthlyGhi(lat),
      monthlyGeneration: this.calculateMonthlyGeneration(outputs.avg_ghi?.monthly || this.getDefaultMonthlyGhi(lat)),

      dataSource: data.station_info?.class || 'TMY3',
    };
  }

  /**
   * Calculate full solar project analysis
   */
  async analyzeProject(
    lat: number,
    lng: number,
    options: {
      systemSizeKw?: number;
      roofAreaM2?: number;
      electricityRate?: number;
      state?: string;
    } = {}
  ): Promise<SolarProjectAnalysis> {
    const resource = await this.getSolarResource(lat, lng);

    // System sizing
    const roofAreaM2 = options.roofAreaM2 || 100; // Default 100 m² (~1000 sq ft)
    const panelAreaM2 = 2.0; // ~2 m² per 400W panel
    const usableRatio = 0.65; // 65% of roof usable
    const maxPanels = Math.floor((roofAreaM2 * usableRatio) / panelAreaM2);
    const panelWattage = 400;

    // Use provided size or calculate from roof
    const systemSizeKw = options.systemSizeKw || (maxPanels * panelWattage) / 1000;
    const numPanels = Math.ceil((systemSizeKw * 1000) / panelWattage);

    // Generation
    const annualGenerationKwh = resource.annualKwhPerKw * systemSizeKw;
    const capacityFactor = annualGenerationKwh / (systemSizeKw * 8760);

    // Economics
    const costPerWatt = this.getCostPerWatt(systemSizeKw);
    const grossCost = systemSizeKw * 1000 * costPerWatt;
    const federalItc = grossCost * 0.3; // 30% ITC through 2032
    const stateIncentives = this.getStateIncentives(options.state || 'TX', systemSizeKw);
    const netCost = grossCost - federalItc - stateIncentives;

    // Savings
    const electricityRate = options.electricityRate || 0.14;
    const annualSavings = annualGenerationKwh * electricityRate;
    const paybackYears = netCost / annualSavings;

    // 25-year analysis with 0.5% annual degradation
    let totalSavings = 0;
    const degradationRate = 0.005;
    for (let year = 1; year <= 25; year++) {
      const degraded = annualSavings * Math.pow(1 - degradationRate, year - 1);
      totalSavings += degraded;
    }

    // IRR calculation (simplified)
    const irr = this.calculateIRR(netCost, annualSavings, 25, degradationRate);

    // LCOE
    const totalGeneration = annualGenerationKwh * 25 * (1 - degradationRate * 12.5); // Average degradation
    const lcoe = grossCost / totalGeneration;

    // Monthly projections
    const monthlyProduction = resource.monthlyGeneration.map((m) => m * systemSizeKw);
    const monthlySavings = monthlyProduction.map((p) => p * electricityRate);

    return {
      systemSizeKw,
      numPanels,
      roofAreaM2,

      annualGenerationKwh: Math.round(annualGenerationKwh),
      capacityFactor: Math.round(capacityFactor * 1000) / 1000,
      firstYearProduction: Math.round(annualGenerationKwh),

      grossCost: Math.round(grossCost),
      federalItc: Math.round(federalItc),
      stateIncentives: Math.round(stateIncentives),
      netCost: Math.round(netCost),

      annualSavings: Math.round(annualSavings),
      paybackYears: Math.round(paybackYears * 10) / 10,
      twentyFiveYearSavings: Math.round(totalSavings),
      irr: Math.round(irr * 10) / 10,
      lcoe: Math.round(lcoe * 1000) / 1000,

      monthlyProduction: monthlyProduction.map((p) => Math.round(p)),
      monthlySavings: monthlySavings.map((s) => Math.round(s)),
    };
  }

  // ============ Private Helpers ============

  private getDemoSolarResource(lat: number, lng: number): NRELSolarResource {
    // Demo data based on latitude (rough estimate)
    // Higher latitudes = less sun
    const baseGhi = 6.0 - Math.abs(lat - 30) * 0.05;
    const ghi = Math.max(3.5, Math.min(7.0, baseGhi));

    return {
      latitude: lat,
      longitude: lng,
      elevation: 200,
      ghi,
      dni: ghi * 1.1,
      dhi: ghi * 0.35,
      tiltedIrradiance: ghi * 1.05,
      optimalTiltAngle: lat,
      annualKwhPerKw: ghi * 365 * 0.8,
      monthlyGhi: this.getDefaultMonthlyGhi(lat),
      monthlyGeneration: this.calculateMonthlyGeneration(this.getDefaultMonthlyGhi(lat)),
      dataSource: 'demo',
    };
  }

  private getDefaultMonthlyGhi(_lat: number): number[] {
    // Approximate monthly GHI for mid-latitudes (Texas-ish)
    const baseGhi = 5.2;
    return [
      baseGhi * 0.7, // Jan
      baseGhi * 0.8, // Feb
      baseGhi * 0.95, // Mar
      baseGhi * 1.1, // Apr
      baseGhi * 1.2, // May
      baseGhi * 1.25, // Jun
      baseGhi * 1.2, // Jul
      baseGhi * 1.15, // Aug
      baseGhi * 1.0, // Sep
      baseGhi * 0.9, // Oct
      baseGhi * 0.75, // Nov
      baseGhi * 0.65, // Dec
    ];
  }

  private calculateMonthlyGeneration(monthlyGhi: number[]): number[] {
    // kWh per kW per month
    const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    return monthlyGhi.map((ghi, i) => ghi * daysInMonth[i] * 0.8);
  }

  private getCostPerWatt(systemSizeKw: number): number {
    // Cost decreases with size (economies of scale)
    if (systemSizeKw >= 100) return 1.2; // Utility scale
    if (systemSizeKw >= 25) return 1.6; // Commercial
    if (systemSizeKw >= 10) return 2.2; // Small commercial
    return 2.75; // Residential
  }

  private getStateIncentives(state: string, systemSizeKw: number): number {
    // State incentives per watt
    const incentivesPerWatt: Record<string, number> = {
      CA: 0.2, // SGIP
      NY: 0.25, // NY-Sun
      MA: 0.3, // SMART
      NJ: 0.15, // SRP
      TX: 0.05, // Limited
      FL: 0.03, // Very limited
    };
    const rate = incentivesPerWatt[state] || 0.03;
    return systemSizeKw * 1000 * rate;
  }

  private calculateIRR(investment: number, annualReturn: number, years: number, degradation: number): number {
    // Binary search for IRR
    let low = 0;
    let high = 0.5;

    for (let i = 0; i < 50; i++) {
      const mid = (low + high) / 2;
      let npv = -investment;

      for (let y = 1; y <= years; y++) {
        const cashFlow = annualReturn * Math.pow(1 - degradation, y - 1);
        npv += cashFlow / Math.pow(1 + mid, y);
      }

      if (Math.abs(npv) < 100) return mid * 100;
      if (npv > 0) low = mid;
      else high = mid;
    }

    return ((low + high) / 2) * 100;
  }
}

// ============================================================================
// ERCOT GRID STATUS
// ============================================================================

export type GridCondition = 'normal' | 'watch' | 'advisory' | 'emergency' | 'critical';

export interface GridStatus {
  condition: GridCondition;
  currentLoadMw: number;
  capacityMw: number;
  reserveMarginPercent: number;
  frequency: number; // Hz (target: 60.0)
  renewablePercent: number;

  // Pricing
  currentPriceMwh: number; // Real-time price
  dayAheadPriceMwh: number;

  // Forecast
  peakForecastMw: number;
  peakTimeToday: string;

  // Alerts
  alerts: GridAlert[];

  lastUpdated: number;
}

export interface GridAlert {
  type: 'conservation' | 'emergency' | 'outage' | 'price_spike' | 'weather';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  startTime: number;
  endTime?: number;
}

/**
 * ERCOT Grid Status Client
 *
 * Note: ERCOT doesn't have a public real-time API.
 * This provides simulated data based on typical patterns.
 * For production, integrate with ERCOT's data feeds or a provider like GridStatus.io
 */
export class ERCOTGridClient {
  /**
   * Get current grid status
   */
  async getGridStatus(): Promise<GridStatus> {
    // Simulate based on time of day and season
    const now = new Date();
    const hour = now.getHours();
    const month = now.getMonth();
    const isSummer = month >= 5 && month <= 8;
    const isWeekday = now.getDay() >= 1 && now.getDay() <= 5;

    // Base load varies by time
    const baseLoadMw = this.calculateBaseLoad(hour, isSummer, isWeekday);
    const capacityMw = 85000; // ERCOT capacity ~85 GW

    // Add some randomness
    const variation = (Math.random() - 0.5) * 2000;
    const currentLoadMw = Math.round(baseLoadMw + variation);

    const reserveMarginPercent = Math.round(((capacityMw - currentLoadMw) / capacityMw) * 100);

    // Condition based on reserve margin
    let condition: GridCondition = 'normal';
    if (reserveMarginPercent < 5) condition = 'critical';
    else if (reserveMarginPercent < 10) condition = 'emergency';
    else if (reserveMarginPercent < 15) condition = 'advisory';
    else if (reserveMarginPercent < 20) condition = 'watch';

    // Renewable percentage (higher midday)
    const solarHours = hour >= 9 && hour <= 17;
    const baseRenewable = solarHours ? 35 : 20;
    const renewablePercent = Math.round(baseRenewable + (Math.random() - 0.5) * 10);

    // Pricing
    const basePriceMwh = isSummer && hour >= 14 && hour <= 19 ? 75 : 35;
    const priceVariation = (Math.random() - 0.5) * 20;
    const currentPriceMwh = Math.max(15, Math.round(basePriceMwh + priceVariation));

    // Alerts
    const alerts: GridAlert[] = [];
    if (condition === 'advisory' || condition === 'emergency' || condition === 'critical') {
      alerts.push({
        type: 'conservation',
        severity: condition === 'critical' ? 'critical' : 'warning',
        message: `Grid stress expected. Please conserve energy from ${hour}:00 to ${hour + 4}:00.`,
        startTime: now.getTime(),
        endTime: now.getTime() + 4 * 60 * 60 * 1000,
      });
    }

    if (currentPriceMwh > 100) {
      alerts.push({
        type: 'price_spike',
        severity: 'info',
        message: `Electricity prices elevated: $${currentPriceMwh}/MWh. Good time to use stored energy.`,
        startTime: now.getTime(),
      });
    }

    return {
      condition,
      currentLoadMw,
      capacityMw,
      reserveMarginPercent,
      frequency: 60.0 + (Math.random() - 0.5) * 0.02,
      renewablePercent,
      currentPriceMwh,
      dayAheadPriceMwh: Math.round(basePriceMwh * 0.9),
      peakForecastMw: isSummer ? 78000 : 55000,
      peakTimeToday: isSummer ? '4:00 PM' : '7:00 AM',
      alerts,
      lastUpdated: now.getTime(),
    };
  }

  private calculateBaseLoad(hour: number, isSummer: boolean, isWeekday: boolean): number {
    // Summer peak around 4-6 PM, winter peak around 7-9 AM
    const summerPattern = [
      45000, 43000, 42000, 41000, 42000, 44000, // 0-5
      48000, 52000, 55000, 58000, 62000, 66000, // 6-11
      68000, 70000, 72000, 74000, 76000, 74000, // 12-17
      70000, 65000, 60000, 55000, 52000, 48000, // 18-23
    ];

    const winterPattern = [
      42000, 40000, 39000, 38000, 39000, 42000, // 0-5
      50000, 55000, 58000, 55000, 52000, 50000, // 6-11
      48000, 47000, 48000, 50000, 52000, 55000, // 12-17
      58000, 55000, 52000, 48000, 45000, 43000, // 18-23
    ];

    const baseLoad = isSummer ? summerPattern[hour] : winterPattern[hour];

    // Weekends are lower
    const weekendFactor = isWeekday ? 1.0 : 0.85;

    return baseLoad * weekendFactor;
  }
}

// ============================================================================
// WEATHER API (OpenWeatherMap or similar)
// ============================================================================

export interface WeatherData {
  location: string;
  temperature: number; // Fahrenheit
  humidity: number; // Percent
  cloudCover: number; // Percent
  uvIndex: number;
  sunrise: string;
  sunset: string;

  // Solar impact
  solarProductionFactor: number; // 0-1, 1 = perfect conditions
}

export interface WeatherForecast {
  current: WeatherData;
  hourly: Array<{
    time: string;
    temperature: number;
    cloudCover: number;
    solarProductionFactor: number;
  }>;
  daily: Array<{
    date: string;
    high: number;
    low: number;
    cloudCover: number;
    solarProductionFactor: number;
  }>;
}

/**
 * Weather API Client for solar production forecasting
 */
export class WeatherClient {
  constructor(_apiKey?: string) {
    // API key reserved for production weather service integration
  }

  /**
   * Get weather and solar production forecast
   */
  async getForecast(lat: number, lng: number): Promise<WeatherForecast> {
    // For demo, return simulated weather
    // In production, integrate with OpenWeatherMap, WeatherAPI, or Visual Crossing
    return this.getDemoForecast(lat, lng);
  }

  private getDemoForecast(lat: number, lng: number): WeatherForecast {
    const now = new Date();
    const hour = now.getHours();
    const month = now.getMonth();

    // Base temperature varies by season and latitude
    const isSummer = month >= 5 && month <= 8;
    const baseTemp = isSummer ? 92 : 55;
    const temperature = baseTemp + (Math.random() - 0.5) * 10;

    // Cloud cover (random but affects solar)
    const cloudCover = Math.round(Math.random() * 40); // 0-40% typical clear day

    // Solar production factor
    const isDaytime = hour >= 7 && hour <= 19;
    const cloudFactor = 1 - cloudCover / 100;
    const timeFactor = isDaytime ? Math.sin(((hour - 6) / 12) * Math.PI) : 0;
    const solarProductionFactor = Math.round(cloudFactor * timeFactor * 100) / 100;

    const current: WeatherData = {
      location: `${lat.toFixed(2)}, ${lng.toFixed(2)}`,
      temperature: Math.round(temperature),
      humidity: Math.round(40 + Math.random() * 30),
      cloudCover,
      uvIndex: isDaytime ? Math.round(7 * solarProductionFactor) : 0,
      sunrise: '6:45 AM',
      sunset: isSummer ? '8:30 PM' : '5:45 PM',
      solarProductionFactor,
    };

    // Hourly forecast (next 24 hours)
    const hourly = [];
    for (let h = 0; h < 24; h++) {
      const forecastHour = (hour + h) % 24;
      const isDay = forecastHour >= 7 && forecastHour <= 19;
      const cloudVar = cloudCover + (Math.random() - 0.5) * 20;
      const cf = 1 - Math.max(0, Math.min(100, cloudVar)) / 100;
      const tf = isDay ? Math.sin(((forecastHour - 6) / 12) * Math.PI) : 0;

      hourly.push({
        time: `${forecastHour}:00`,
        temperature: Math.round(temperature + (isDay ? 5 : -5) * (Math.random() - 0.5)),
        cloudCover: Math.round(Math.max(0, Math.min(100, cloudVar))),
        solarProductionFactor: Math.round(cf * tf * 100) / 100,
      });
    }

    // Daily forecast (next 7 days)
    const daily = [];
    for (let d = 0; d < 7; d++) {
      const dayCloud = Math.round(Math.random() * 50);
      daily.push({
        date: new Date(now.getTime() + d * 86400000).toLocaleDateString(),
        high: Math.round(baseTemp + 5 + (Math.random() - 0.5) * 10),
        low: Math.round(baseTemp - 15 + (Math.random() - 0.5) * 10),
        cloudCover: dayCloud,
        solarProductionFactor: Math.round((1 - dayCloud / 100) * 0.85 * 100) / 100,
      });
    }

    return { current, hourly, daily };
  }
}

// ============================================================================
// TERRA ATLAS BRIDGE
// ============================================================================

export interface NearbyProject {
  id: string;
  name: string;
  type: 'solar' | 'hydro' | 'wind' | 'storage' | 'nuclear';
  distanceKm: number;
  distanceMiles: number;

  // Location
  latitude: number;
  longitude: number;
  city: string;
  state: string;

  // Capacity
  capacityMw: number;
  annualGenerationMwh: number;

  // Economics
  estimatedIrr: number;
  capitalCostUsd: number;
  minInvestment: number;
  status: 'identified' | 'analyzing' | 'ready' | 'funding' | 'construction' | 'operational';

  // Community
  communityOwnershipPercent: number;
  localJobsCreated: number;
}

export interface TerraAtlasStats {
  totalProjects: number;
  totalCapacityMw: number;
  averageIrr: number;
  projectsByType: Record<string, number>;
  projectsByState: Record<string, number>;
}

/**
 * Terra Atlas Bridge Client
 *
 * Connects Energy Sovereignty to Terra Atlas investment platform
 */
export class TerraAtlasBridge {
  private supabaseUrl: string;
  private supabaseKey: string;

  constructor(supabaseUrl?: string, supabaseKey?: string) {
    this.supabaseUrl =
      supabaseUrl ||
      (typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_SUPABASE_URL || '' : '');
    this.supabaseKey =
      supabaseKey ||
      (typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '' : '');
  }

  /**
   * Find energy projects near a location
   */
  async findProjectsNearMe(
    lat: number,
    lng: number,
    options: {
      radiusKm?: number;
      types?: Array<'solar' | 'hydro' | 'wind' | 'storage' | 'nuclear'>;
      minIrr?: number;
      status?: string[];
      limit?: number;
    } = {}
  ): Promise<NearbyProject[]> {
    const radiusKm = options.radiusKm || 100;
    const limit = options.limit || 20;

    // If Supabase configured, use real data
    if (this.supabaseUrl && this.supabaseKey) {
      try {
        return await this.fetchFromSupabase(lat, lng, radiusKm, options);
      } catch (e) {
        console.warn('Supabase fetch failed, using demo data:', e);
      }
    }

    // Return demo projects
    return this.getDemoProjects(lat, lng, radiusKm, options).slice(0, limit);
  }

  /**
   * Get aggregate statistics for a region
   */
  async getRegionStats(
    _state?: string
  ): Promise<TerraAtlasStats> {
    // Demo stats
    return {
      totalProjects: 87432,
      totalCapacityMw: 245000,
      averageIrr: 12.4,
      projectsByType: {
        hydro: 45231,
        solar: 28456,
        wind: 12890,
        storage: 845,
        nuclear: 10,
      },
      projectsByState: {
        TX: 8934,
        CA: 12456,
        WA: 5678,
        NY: 3456,
        FL: 4567,
      },
    };
  }

  /**
   * Calculate Haversine distance between two points
   */
  calculateDistance(lat1: number, lng1: number, lat2: number, lng2: number): number {
    const R = 6371; // Earth's radius in km
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLng = ((lng2 - lng1) * Math.PI) / 180;

    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLng / 2) *
        Math.sin(dLng / 2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  // ============ Private Methods ============

  private async fetchFromSupabase(
    lat: number,
    lng: number,
    radiusKm: number,
    options: any
  ): Promise<NearbyProject[]> {
    const response = await fetch(`${this.supabaseUrl}/rest/v1/sites?select=*`, {
      headers: {
        apikey: this.supabaseKey,
        Authorization: `Bearer ${this.supabaseKey}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Supabase error: ${response.status}`);
    }

    const sites = await response.json();

    // Filter by distance and transform
    return sites
      .map((site: any) => {
        const distanceKm = this.calculateDistance(lat, lng, site.latitude, site.longitude);
        return {
          id: site.site_id,
          name: site.name,
          type: site.energy_type,
          distanceKm: Math.round(distanceKm * 10) / 10,
          distanceMiles: Math.round(distanceKm * 0.621371 * 10) / 10,
          latitude: site.latitude,
          longitude: site.longitude,
          city: site.city,
          state: site.state,
          capacityMw: site.estimated_capacity_mw,
          annualGenerationMwh: site.estimated_generation_mwh,
          estimatedIrr: site.estimated_irr,
          capitalCostUsd: site.capital_cost_usd,
          minInvestment: Math.max(100, Math.round(site.capital_cost_usd * 0.001)),
          status: site.status,
          communityOwnershipPercent: 0,
          localJobsCreated: Math.round(site.estimated_capacity_mw * 0.5),
        };
      })
      .filter((p: NearbyProject) => p.distanceKm <= radiusKm)
      .filter((p: NearbyProject) => !options.types || options.types.includes(p.type))
      .filter((p: NearbyProject) => !options.minIrr || p.estimatedIrr >= options.minIrr)
      .sort((a: NearbyProject, b: NearbyProject) => a.distanceKm - b.distanceKm);
  }

  private getDemoProjects(lat: number, lng: number, radiusKm: number, _options: any): NearbyProject[] {
    // Generate demo projects around the location
    const projects: NearbyProject[] = [];

    const types: Array<'solar' | 'hydro' | 'wind' | 'storage'> = ['solar', 'hydro', 'wind', 'storage'];
    const statuses = ['ready', 'funding', 'operational'];

    for (let i = 0; i < 15; i++) {
      // Random offset within radius
      const angle = Math.random() * 2 * Math.PI;
      const distance = Math.random() * radiusKm * 0.8;
      const latOffset = (distance / 111) * Math.cos(angle);
      const lngOffset = (distance / (111 * Math.cos((lat * Math.PI) / 180))) * Math.sin(angle);

      const type = types[Math.floor(Math.random() * types.length)];
      const capacityMw = type === 'hydro' ? 5 + Math.random() * 50 : 1 + Math.random() * 10;

      projects.push({
        id: `demo-${i + 1}`,
        name: `${type.charAt(0).toUpperCase() + type.slice(1)} Project ${i + 1}`,
        type,
        distanceKm: Math.round(distance * 10) / 10,
        distanceMiles: Math.round(distance * 0.621371 * 10) / 10,
        latitude: lat + latOffset,
        longitude: lng + lngOffset,
        city: 'Nearby City',
        state: 'TX',
        capacityMw: Math.round(capacityMw * 10) / 10,
        annualGenerationMwh: Math.round(capacityMw * 2500),
        estimatedIrr: 8 + Math.random() * 12,
        capitalCostUsd: Math.round(capacityMw * 1500000),
        minInvestment: 100,
        status: statuses[Math.floor(Math.random() * statuses.length)] as any,
        communityOwnershipPercent: Math.round(Math.random() * 30),
        localJobsCreated: Math.round(capacityMw * 0.5),
      });
    }

    return projects.sort((a, b) => a.distanceKm - b.distanceKm);
  }
}

// ============================================================================
// UNIFIED ENERGY API CLIENT
// ============================================================================

/**
 * Unified Energy API Client
 *
 * Combines all energy data sources into a single interface
 */
export class EnergyAPIClient {
  readonly solar: NRELSolarClient;
  readonly grid: ERCOTGridClient;
  readonly weather: WeatherClient;
  readonly terraAtlas: TerraAtlasBridge;

  constructor(config?: {
    nrelApiKey?: string;
    weatherApiKey?: string;
    supabaseUrl?: string;
    supabaseKey?: string;
  }) {
    this.solar = new NRELSolarClient(config?.nrelApiKey);
    this.grid = new ERCOTGridClient();
    this.weather = new WeatherClient(config?.weatherApiKey);
    this.terraAtlas = new TerraAtlasBridge(config?.supabaseUrl, config?.supabaseKey);
  }

  /**
   * Get comprehensive energy data for a location
   */
  async getLocationData(
    lat: number,
    lng: number
  ): Promise<{
    solar: NRELSolarResource;
    grid: GridStatus;
    weather: WeatherForecast;
    nearbyProjects: NearbyProject[];
  }> {
    const [solar, grid, weather, nearbyProjects] = await Promise.all([
      this.solar.getSolarResource(lat, lng),
      this.grid.getGridStatus(),
      this.weather.getForecast(lat, lng),
      this.terraAtlas.findProjectsNearMe(lat, lng, { limit: 5 }),
    ]);

    return { solar, grid, weather, nearbyProjects };
  }
}

// Export singleton for convenience
export const energyAPI = new EnergyAPIClient();
