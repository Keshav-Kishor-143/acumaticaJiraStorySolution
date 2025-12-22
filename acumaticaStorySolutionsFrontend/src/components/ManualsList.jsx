import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
  useTheme,
  alpha,
} from '@mui/material';
import {
  MenuBook as MenuBookIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { getManualsList } from '../api/solutions';

const ManualsList = () => {
  const theme = useTheme();
  const [manuals, setManuals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchManuals = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getManualsList();
      if (response.success) {
        setManuals(response.manuals || []);
      } else {
        setError(response.error || 'Failed to load manuals');
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch manuals list');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchManuals();
  }, []);

  return (
    <Paper
      elevation={2}
      sx={{
        height: '100%',
        maxHeight: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 2,
        overflow: 'hidden',
        bgcolor: theme.palette.background.paper,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: `1px solid ${theme.palette.divider}`,
          bgcolor: alpha(theme.palette.primary.main, 0.05),
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
        }}
      >
        <Box
          sx={{
            width: 36,
            height: 36,
            borderRadius: 1.5,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.primary.dark})`,
            color: 'white',
          }}
        >
          <MenuBookIcon sx={{ fontSize: 20 }} />
        </Box>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1rem' }}>
            Knowledge Base
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Available Manuals
          </Typography>
        </Box>
        {!loading && (
          <Box
            onClick={fetchManuals}
            sx={{
              cursor: 'pointer',
              p: 0.5,
              borderRadius: 1,
              display: 'flex',
              alignItems: 'center',
              color: 'text.secondary',
              transition: 'all 0.2s',
              '&:hover': {
                bgcolor: 'action.hover',
                color: 'primary.main',
              },
            }}
            title="Refresh list"
          >
            <RefreshIcon sx={{ fontSize: 18 }} />
          </Box>
        )}
      </Box>

      {/* Content */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          p: 1,
          minHeight: 0, // Important for flex scrolling
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'transparent',
          },
          '&::-webkit-scrollbar-thumb': {
            background: alpha(theme.palette.primary.main, 0.2),
            borderRadius: '4px',
            '&:hover': {
              background: alpha(theme.palette.primary.main, 0.3),
            },
          },
        }}
      >
        {loading && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 4,
              gap: 2,
            }}
          >
            <CircularProgress size={32} />
            <Typography variant="body2" color="text.secondary">
              Loading manuals...
            </Typography>
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ m: 1, borderRadius: 1.5 }}>
            {error}
          </Alert>
        )}

        {!loading && !error && manuals.length === 0 && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 4,
              px: 2,
            }}
          >
            <MenuBookIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
            <Typography variant="body2" color="text.secondary" align="center">
              No manuals found in knowledge base
            </Typography>
          </Box>
        )}

        {!loading && !error && manuals.length > 0 && (
          <List sx={{ py: 0 }}>
            {manuals.map((manual, index) => (
              <ListItem
                key={manual.name || index}
                sx={{
                  borderRadius: 1.5,
                  mb: 0.5,
                  transition: 'all 0.2s',
                  '&:hover': {
                    bgcolor: alpha(theme.palette.primary.main, 0.08),
                    transform: 'translateX(4px)',
                  },
                }}
              >
                <ListItemText
                  primary={
                    <Typography
                      variant="body2"
                      sx={{
                        fontWeight: 500,
                        color: 'text.primary',
                        fontSize: '0.875rem',
                        lineHeight: 1.5,
                      }}
                    >
                      {manual.display_name || manual.name}
                    </Typography>
                  }
                  secondary={
                    <Typography
                      variant="caption"
                      sx={{
                        color: 'text.secondary',
                        fontSize: '0.75rem',
                        mt: 0.25,
                        display: 'block',
                      }}
                    >
                      {manual.name}
                    </Typography>
                  }
                />
              </ListItem>
            ))}
          </List>
        )}
      </Box>

      {/* Footer */}
      {!loading && !error && manuals.length > 0 && (
        <Box
          sx={{
            p: 1.5,
            borderTop: `1px solid ${theme.palette.divider}`,
            bgcolor: alpha(theme.palette.primary.main, 0.03),
          }}
        >
          <Typography variant="caption" color="text.secondary" align="center" sx={{ display: 'block' }}>
            {manuals.length} {manuals.length === 1 ? 'manual' : 'manuals'} available
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default ManualsList;

