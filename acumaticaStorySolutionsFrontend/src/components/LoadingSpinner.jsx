import React from 'react';
import { Box, Typography, useTheme, alpha, CircularProgress } from '@mui/material';
import { AutoAwesome as SparkleIcon } from '@mui/icons-material';

const LoadingSpinner = ({ message = 'Processing...', progress = null }) => {
  const theme = useTheme();

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 3,
        padding: 6,
        borderRadius: 3,
        background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)}, ${alpha(theme.palette.secondary?.main || theme.palette.primary.main, 0.05)})`,
        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
      }}
    >
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        <CircularProgress
          size={64}
          thickness={4}
          sx={{
            color: theme.palette.primary.main,
            animationDuration: '2s',
          }}
        />
        <Box
          sx={{
            top: 0,
            left: 0,
            bottom: 0,
            right: 0,
            position: 'absolute',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <SparkleIcon
            sx={{
              fontSize: 32,
              color: theme.palette.primary.main,
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%, 100%': { opacity: 1, transform: 'scale(1)' },
                '50%': { opacity: 0.7, transform: 'scale(1.1)' },
              },
            }}
          />
        </Box>
      </Box>
      <Box sx={{ textAlign: 'center' }}>
        <Typography 
          variant="h6" 
          sx={{ 
            fontWeight: 600,
            mb: 1,
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || theme.palette.primary.main})`,
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          {message}
        </Typography>
        {progress && (
          <Typography variant="body2" color="text.secondary">
            {progress.message || 'Please wait...'}
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default LoadingSpinner;

