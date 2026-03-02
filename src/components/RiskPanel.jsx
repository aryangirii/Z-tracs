import React from 'react';
import './RiskPanel.css';

const RiskPanel = ({ risk }) => {
  return (
    <div className="risk-panel">
      <h3>Risk Level</h3>
      <p>{risk || 'Unknown'}</p>
    </div>
  );
};

export default RiskPanel;
