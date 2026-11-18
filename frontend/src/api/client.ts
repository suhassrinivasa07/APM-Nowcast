export interface HealthData {
  status: string;
  service: string;
  version: string;
  timestamp: string;
  environment: string;
  device_target?: string | null;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1';
const ROOT_URL = API_BASE.includes('/api/v1')
  ? API_BASE.replace(/\/api\/v1\/?$/, '')
  : API_BASE;

export async function fetchHealth(): Promise<HealthData> {
  const response = await fetch(`${ROOT_URL || ''}/health`, {
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }

  return response.json();
}
