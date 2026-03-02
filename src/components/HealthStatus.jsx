import React from 'react';
import './HealthStatus.css';

const HealthStatus = ({ modelLoaded, version }) => {
  return (
    <div className="health-status">
      <span>Model Loaded: {modelLoaded ? 'YES' : 'NO'}</span>
      <span>Version: {version || 'n/a'}</span>
    </div>
  );
};

export default HealthStatus;
