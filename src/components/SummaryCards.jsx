import React from 'react';
import './SummaryCards.css';

const SummaryCards = ({ data }) => {
  return (
    <div className="summary-cards">
      {/* placeholder cards */}
      <div className="card">Average Congestion: {data?.average || '--'}</div>
      <div className="card">Risk Level: {data?.risk || '--'}</div>
      <div className="card">Next Peak: {data?.peak || '--'}</div>
    </div>
  );
};

export default SummaryCards;
