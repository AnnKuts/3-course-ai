export default function ProbabilityList({ probabilities, predicted }) {
  return (
    <ol className="probs" aria-label="Ймовірності для цифр 0–9">
      {probabilities.map((value, digit) => {
        const width = Math.min(100, Math.max(0, value));
        return (
          <li key={digit} className={digit === predicted ? 'prob prob-best' : 'prob'}>
            <span className="prob-digit">{digit}</span>
            <span className="prob-track" aria-hidden="true">
              <span className="prob-fill" style={{ width: `${width}%` }} />
            </span>
            <span className="prob-value">{value.toFixed(1)}%</span>
          </li>
        );
      })}
    </ol>
  );
}
