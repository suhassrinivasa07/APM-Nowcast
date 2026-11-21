export interface HealthData {
  status: string;
  service: string;
  version: string;
  timestamp: string;
  environment: string;
  device_target?: string | null;
}

const RAW_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const API_BASE = RAW_BASE.replace(/\/+$/, '');
const ROOT_URL = API_BASE.includes('/api/v1')
  ? API_BASE.replace(/\/api\/v1\/?$/, '')
  : API_BASE;

export async function fetchHealth(): Promise<HealthData> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3500);

  try {
    const response = await fetch(`${ROOT_URL}/health`, {
      headers: {
        'Accept': 'application/json',
      },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Health check failed with status: ${response.status}`);
    }

    return response.json();
  } catch (err) {
    clearTimeout(timeoutId);
    throw err;
  }
}
