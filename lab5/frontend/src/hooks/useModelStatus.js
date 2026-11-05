import { useCallback, useEffect, useState } from 'react';
import { fetchStatus } from '../api.js';

export function useModelStatus() {
  const [status, setStatus] = useState('checking');

  const check = useCallback(async (signal) => {
    try {
      const trained = await fetchStatus(signal);
      setStatus(trained ? 'ready' : 'untrained');
    } catch (error) {
      if (error?.name !== 'AbortError') setStatus('offline');
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    const controller = new AbortController();

    async function initialCheck() {
      try {
        const trained = await fetchStatus(controller.signal);
        if (!ignore) {
          setStatus(trained ? 'ready' : 'untrained');
        }
      } catch (error) {
        if (!ignore && error?.name !== 'AbortError') {
          setStatus('offline');
        }
      }
    }

    initialCheck();

    const onFocus = () => {
      check();
    };
    window.addEventListener('focus', onFocus);

    return () => {
      ignore = true;
      controller.abort();
      window.removeEventListener('focus', onFocus);
    };
  }, [check]);

  return { status, recheck: check };
}
