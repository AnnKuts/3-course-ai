import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPencil } from '@fortawesome/free-solid-svg-icons';

const SIZE = 560;
const BACKGROUND = '#09090b';
const STROKE = '#ffffff';
const LINE_WIDTH = 42;

const DrawingCanvas = forwardRef(function DrawingCanvas({ isEmpty, onStroke }, ref) {
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);

  const getContext = () => canvasRef.current.getContext('2d', { willReadFrequently: true });

  const clear = useCallback(() => {
    const ctx = getContext();
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = LINE_WIDTH;
    ctx.strokeStyle = STROKE;
    drawingRef.current = false;
  }, []);

  useImperativeHandle(ref, () => ({
    clear,
    toDataURL: () => canvasRef.current.toDataURL('image/png'),
  }), [clear]);

  useEffect(() => { clear(); }, [clear]);

  const toCanvasPoint = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (SIZE / rect.width),
      y: (event.clientY - rect.top) * (SIZE / rect.height),
    };
  };

  const handlePointerDown = (event) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    event.preventDefault();
    drawingRef.current = true;
    canvasRef.current.setPointerCapture(event.pointerId);
    const ctx = getContext();
    const { x, y } = toCanvasPoint(event);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01);
    ctx.stroke();
    onStroke?.();
  };

  const handlePointerMove = (event) => {
    if (!drawingRef.current) return;
    const ctx = getContext();
    const { x, y } = toCanvasPoint(event);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stop = () => { drawingRef.current = false; };

  return (
    <div className="canvas-frame">
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        className="canvas"
        aria-label="Поле для малювання цифри від 0 до 9"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={stop}
        onPointerCancel={stop}
        onLostPointerCapture={stop}
      />
      {isEmpty && (
        <div className="canvas-placeholder" aria-hidden="true">
          <FontAwesomeIcon icon={faPencil} />
          <span>Намалюйте цифру</span>
        </div>
      )}
    </div>
  );
});

export default DrawingCanvas;
