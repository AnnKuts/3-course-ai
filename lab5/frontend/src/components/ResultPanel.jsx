import ProbabilityList from './ProbabilityList.jsx';

function confidenceLevel(value) {
  if (value >= 90) return { key: 'high', label: 'висока впевненість' };
  if (value >= 60) return { key: 'medium', label: 'середня впевненість' };
  return { key: 'low', label: 'низька впевненість' };
}

export default function ResultPanel({ result, isLoading, isStale }) {
  if (!result) {
    return (
      <div className="empty" aria-busy={isLoading}>
        <p className="empty-title">{isLoading ? 'Розпізнавання…' : 'Немає прогнозу'}</p>
        <p className="empty-text">Намалюйте цифру й натисніть «Розпізнати»</p>
      </div>
    );
  }

  const level = confidenceLevel(result.confidence);

  return (
    <div className={`result${isLoading ? ' is-loading' : ''}`} aria-busy={isLoading}>
      <div className="summary">
        <div className="summary-digit" aria-label={`Розпізнана цифра ${result.digit}`}>
          {result.digit}
        </div>
        <div className="summary-meta">
          <span className="meta-label">Прогноз</span>
          <span className="summary-confidence">{result.confidence.toFixed(1)}%</span>
          <span className={`level level-${level.key}`}>{level.label}</span>
        </div>
        <figure className="preview">
          <img src={result.processed} alt="Вхід мережі" />
          <figcaption>28×28</figcaption>
        </figure>
      </div>

      {isStale && !isLoading && (
        <p className="stale-note">Малюнок змінено</p>
      )}

      <div className="section">
        <div className="section-head">
          <span className="meta-label">Розподіл softmax</span>
        </div>
        <ProbabilityList probabilities={result.probabilities} predicted={result.digit} />
      </div>
    </div>
  );
}
