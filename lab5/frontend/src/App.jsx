import { useCallback, useEffect, useRef, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faArrowRight,
  faCircleExclamation,
  faCircleNotch,
  faRotateLeft,
} from '@fortawesome/free-solid-svg-icons';
import DrawingCanvas from './components/DrawingCanvas.jsx';
import ResultPanel from './components/ResultPanel.jsx';
import StatusBadge from './components/StatusBadge.jsx';
import { predictDigit } from './api.js';
import { useModelStatus } from './hooks/useModelStatus.js';

export default function App() {
  const canvasRef = useRef(null);
  const requestRef = useRef(null);
  const { status } = useModelStatus();

  const [hasMarks, setHasMarks] = useState(false);
  const [isStale, setIsStale] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const modelReady = status === 'ready';
  const canPredict = modelReady && !isLoading;

  const handleStroke = useCallback(() => {
    setHasMarks(true);
    setIsStale(true);
    setError('');
  }, []);

  const handleClear = useCallback(() => {
    requestRef.current?.abort();
    canvasRef.current?.clear();
    setHasMarks(false);
    setIsStale(false);
    setIsLoading(false);
    setResult(null);
    setError('');
  }, []);

  const handlePredict = useCallback(async () => {
    if (!canPredict) return;
    if (!hasMarks) {
      setError('Спочатку намалюйте цифру.');
      return;
    }

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;

    setIsLoading(true);
    setError('');
    try {
      const data = await predictDigit(canvasRef.current.toDataURL(), controller.signal);
      setResult(data);
      setIsStale(false);
    } catch (err) {
      if (err.name === 'AbortError') return;
      setError(err.message || 'Не вдалося розпізнати цифру.');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsLoading(false);
      }
    }
  }, [canPredict, hasMarks]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
      const tag = event.target.tagName;
      if (tag === 'BUTTON' || tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (event.key === 'Enter') {
        event.preventDefault();
        handlePredict();
      } else if (event.key === 'Escape') {
        handleClear();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handlePredict, handleClear]);

  useEffect(() => () => requestRef.current?.abort(), []);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1 className="brand-name">Лабораторна 5</h1>
        </div>
        <StatusBadge status={status} />
      </header>

      <main className="workspace">
        <section className="panel" aria-labelledby="draw-title">
          <div className="panel-head">
            <h2 id="draw-title" className="panel-title">Полотно</h2>
            <span className="panel-hint">одна цифра, по центру</span>
          </div>
          <div className="panel-body">
            <DrawingCanvas ref={canvasRef} isEmpty={!hasMarks} onStroke={handleStroke} />

            <div className="actions">
              <button type="button" className="btn btn-secondary" onClick={handleClear}>
                <FontAwesomeIcon icon={faRotateLeft} />
                <span>Очистити</span>
                <kbd>Esc</kbd>
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={!canPredict}
              >
                <FontAwesomeIcon icon={isLoading ? faCircleNotch : faArrowRight} spin={isLoading} />
                <span>{isLoading ? 'Розпізнавання…' : 'Розпізнати'}</span>
                {!isLoading && <kbd>Enter</kbd>}
              </button>
            </div>

            {error && (
              <div className="alert" role="alert">
                <FontAwesomeIcon icon={faCircleExclamation} />
                <span>{error}</span>
              </div>
            )}
            {status === 'untrained' && (
              <p className="notice">
                Ваги моделі не знайдено. Виконайте <code>python train.py</code> і поверніться на сторінку.
              </p>
            )}
          </div>
        </section>

        <section className="panel" aria-labelledby="result-title">
          <div className="panel-head">
            <h2 id="result-title" className="panel-title">Результат</h2>
          </div>
          <div className="panel-body">
            <ResultPanel result={result} isLoading={isLoading} isStale={isStale} />
          </div>
        </section>
      </main>
    </div>
  );
}
