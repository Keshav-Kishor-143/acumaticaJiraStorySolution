import React, { useState, useEffect } from 'react';
import { Chip, Box, Tooltip } from '@mui/material';
import { CheckCircle, Error, Warning } from '@mui/icons-material';
import { healthCheck } from '../api/solutions';

const HealthStatus = () => {
  const [status, setStatus] = useState('checking');
  const [message, setMessage] = useState('Checking...');
  const [components, setComponents] = useState(null);

  useEffect(() => {
    let intervalId = null;
    let checkCount = 0;
    
    const checkHealth = async () => {
      try {
        const response = await healthCheck();
        checkCount++;
        
        // Store component details for tooltip
        if (response.components) {
          setComponents(response.components);
        }
        
        // Handle different status values from backend
        const backendStatus = response.status?.toLowerCase();
        if (backendStatus === 'healthy') {
          setStatus('healthy');
          setMessage(response.message || 'Service available');
          
          // After initial checks, reduce frequency when healthy
          // First 3 checks: every 30s, then every 2 minutes when healthy
          if (checkCount >= 3 && intervalId) {
            clearInterval(intervalId);
            intervalId = setInterval(checkHealth, 120000); // 2 minutes when healthy
          }
        } else if (backendStatus === 'degraded') {
          setStatus('degraded');
          // Increase frequency when degraded (check every 30s)
          if (intervalId) {
            clearInterval(intervalId);
            intervalId = setInterval(checkHealth, 30000);
          }
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
          // Increase frequency when unhealthy (check every 15s)
          if (intervalId) {
            clearInterval(intervalId);
            intervalId = setInterval(checkHealth, 15000);
          }
        } else {
          // If status is not recognized, check if response is successful
          // Some backends might return status in different format
          console.warn('Unknown health status:', backendStatus, 'Full response:', response);
          setStatus('healthy');
          setMessage(response.message || 'Service available');
        }
      } catch (error) {
        console.error('Health check error:', error);
        setStatus('unhealthy');
        setMessage('Service unavailable');
        // Increase frequency on error (check every 15s)
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = setInterval(checkHealth, 15000);
        }
      }
    };

    // Initial check immediately
    checkHealth();
    // Then check every 30 seconds initially
    intervalId = setInterval(checkHealth, 30000);

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
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

