async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function fetchStatus(signal) {
  const response = await fetch('/status', { signal });
  if (!response.ok) throw new Error('Сервер недоступний');
  const data = await readJson(response);
  return Boolean(data?.trained);
}

export async function predictDigit(image, signal) {
  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image }),
    signal,
  });
  const data = await readJson(response);
  if (!response.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail : null;
    throw new Error(detail || `Помилка сервера (${response.status})`);
  }
  if (!data || !Array.isArray(data.probabilities)) {
    throw new Error('Неочікувана відповідь сервера');
  }
  return data;
}
