import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faCircleCheck,
  faCircleExclamation,
  faCircleNotch,
  faTriangleExclamation,
} from '@fortawesome/free-solid-svg-icons';

const STATES = {
  checking: { icon: faCircleNotch, spin: true, label: 'Перевірка моделі…', title: '' },
  ready: { icon: faCircleCheck, label: 'Модель готова', title: 'Ваги model/mnist_cnn.pt завантажено' },
  untrained: { icon: faTriangleExclamation, label: 'Модель не навчена', title: 'Виконайте: python train.py' },
  offline: { icon: faCircleExclamation, label: 'Сервер недоступний', title: 'Запустіть: python -m uvicorn app:app' },
};

export default function StatusBadge({ status }) {
  const s = STATES[status];
  return (
    <div className={`status status-${status}`} role="status" aria-live="polite" title={s.title}>
      <FontAwesomeIcon icon={s.icon} spin={s.spin} />
      <span>{s.label}</span>
    </div>
  );
}
