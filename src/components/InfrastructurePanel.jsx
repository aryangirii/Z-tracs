import React from 'react';
import './InfrastructurePanel.css';

const InfrastructurePanel = ({ score }) => {
  return (
    <div className="infrastructure-panel">
      <h3>Infrastructure Stress Index</h3>
      <p>{score || 'N/A'}</p>
    </div>
  );
};

export default InfrastructurePanel;
