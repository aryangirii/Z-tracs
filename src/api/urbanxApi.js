// simple fetch wrappers for backend API
const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export async function predict(sequence) {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
  if (!response.ok) throw new Error(`Predict failed: ${response.statusText}`);
  return response.json();
}

export async function simulateScenario(sequence, shockParams) {
  const response = await fetch(`${BASE_URL}/scenario`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence, shock_params: shockParams }),
  });
  if (!response.ok) throw new Error(`Scenario failed: ${response.statusText}`);
  return response.json();
}
