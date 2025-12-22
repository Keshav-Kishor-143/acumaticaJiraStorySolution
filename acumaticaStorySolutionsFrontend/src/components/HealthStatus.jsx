import React, { useState, useEffect } from 'react';
import { Chip, Box, Tooltip } from '@mui/material';
import { CheckCircle, Error, Warning } from '@mui/icons-material';
import { healthCheck } from '../api/solutions';

const HealthStatus = () => {
  const [status, setStatus] = useState('checking');
  const [message, setMessage] = useState('Checking...');
  const [components, setComponents] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await healthCheck();
        console.log('Health check response:', response); // Debug log
        
        // Store component details for tooltip
        if (response.components) {
          setComponents(response.components);
        }
        
        // Handle different status values from backend
        const backendStatus = response.status?.toLowerCase();
        if (backendStatus === 'healthy') {
          setStatus('healthy');
          setMessage(response.message || 'Service available');
        } else if (backendStatus === 'degraded') {
          setStatus('degraded');
          // Build detailed message from components if available
          if (response.components) {
            const failedComponents = Object.entries(response.components)
              .filter(([_, comp]) => comp.status === 'error')
              .map(([name, comp]) => `${name}: ${comp.message || 'error'}`);
            
            if (failedComponents.length > 0) {
              setMessage(`Some components degraded: ${failedComponents.join('; ')}`);
            } else {
              setMessage(response.message || 'Service degraded');
            }
          } else {
            setMessage(response.message || 'Service degraded');
          }
        } else if (backendStatus === 'unhealthy') {
          setStatus('unhealthy');
          setMessage(response.message || 'Service unavailable');
        } else {
          // If status is not recognized, check if response is successful
          // Some backends might return status in different format
          console.warn('Unknown health status:', backendStatus, 'Full response:', response);
          setStatus('healthy');
          setMessage(response.message || 'Service available');
        }
      } catch (error) {
        console.error('Health check error:', error); // Debug log
        setStatus('unhealthy');
        setMessage('Service unavailable');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = () => {
    switch (status) {
      case 'healthy':
        return <CheckCircle fontSize="small" />;
      case 'degraded':
        return <Warning fontSize="small" />;
      default:
        return <Error fontSize="small" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      default:
        return 'error';
    }
  };

  // Build detailed tooltip text with component information
  const getTooltipText = () => {
    if (!components) return message;
    
    const componentDetails = Object.entries(components)
      .map(([name, comp]) => {
        const statusIcon = comp.status === 'ok' ? '✓' : '✗';
        return `${name}: ${statusIcon} ${comp.message || comp.status}`;
      })
      .join('\n');
    
    return `${message}\n\nComponents:\n${componentDetails}`;
  };

  return (
    <Tooltip 
      title={getTooltipText()} 
      arrow
      componentsProps={{
        tooltip: {
          sx: {
            maxWidth: 400,
            whiteSpace: 'pre-line',
            fontSize: '0.75rem',
          }
        }
      }}
    >
      <Chip
        icon={getStatusIcon()}
        label={status === 'checking' ? 'Checking...' : status === 'healthy' ? 'System Online' : 'System Degraded'}
        color={getStatusColor()}
        size="small"
        sx={{
          '& .MuiChip-label': { px: 2 },
        }}
      />
    </Tooltip>
  );
};

export default HealthStatus;

