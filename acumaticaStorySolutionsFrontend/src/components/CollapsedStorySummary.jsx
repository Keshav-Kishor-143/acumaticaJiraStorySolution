import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Chip,
  IconButton,
  Tooltip,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Edit as EditIcon,
  Description as DescriptionIcon,
  Label as LabelIcon,
  CheckCircleOutline as CheckCircleIcon,
  Add as AddIcon,
} from '@mui/icons-material';

const CollapsedStorySummary = ({ storyData, onExpand, onReset, isSolutionActive = false }) => {
  const theme = useTheme();
  
  if (!storyData) return null;
  
  return (
    <Paper
      elevation={2}
      sx={{
        p: 2,
        borderRadius: 2,
        background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.08)}, ${alpha(theme.palette.secondary?.main || theme.palette.primary.main, 0.05)})`,
        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
        transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
          boxShadow: theme.shadows[4],
          transform: 'translateY(-2px)',
        },
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CheckCircleIcon sx={{ fontSize: 18, color: 'success.main' }} />
          <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'primary.main' }}>
            Story Submitted
          </Typography>
        </Box>
        <Tooltip title="Edit Story">
          <IconButton 
            size="small" 
            onClick={onExpand}
            sx={{
              color: 'primary.main',
              '&:hover': {
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                transform: 'rotate(90deg)',
              },
              transition: 'all 0.3s ease',
            }}
          >
            <EditIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      
      {/* Story ID */}
      {storyData.story_id && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <LabelIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
          <Chip
            label={storyData.story_id}
            size="small"
            sx={{
              height: 22,
              fontSize: '0.7rem',
              fontWeight: 600,
              bgcolor: alpha(theme.palette.primary.main, 0.1),
              color: 'primary.main',
            }}
          />
        </Box>
      )}
      
      {/* Title */}
      {storyData.title && (
        <Typography 
          variant="body2" 
          sx={{ 
            fontWeight: 600, 
            mb: 1,
            color: 'text.primary',
            lineHeight: 1.4,
            fontSize: '0.875rem',
          }}
        >
          {storyData.title}
        </Typography>
      )}
      
      {/* Description Preview */}
      {storyData.description && (
        <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
          <DescriptionIcon sx={{ fontSize: 14, color: 'text.secondary', mt: 0.25, flexShrink: 0 }} />
          <Typography 
            variant="caption" 
            sx={{ 
              color: 'text.secondary',
              lineHeight: 1.5,
              display: '-webkit-box',
              WebkitLineClamp: 3,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
              fontSize: '0.75rem',
            }}
          >
            {storyData.description.length > 100 
              ? `${storyData.description.substring(0, 100)}...` 
              : storyData.description}
          </Typography>
        </Box>
      )}
      
      {/* Acceptance Criteria Count */}
      {storyData.acceptance_criteria && storyData.acceptance_criteria.length > 0 && (
        <Chip
          label={`${storyData.acceptance_criteria.length} ${storyData.acceptance_criteria.length === 1 ? 'criterion' : 'criteria'}`}
          size="small"
          variant="outlined"
          sx={{
            fontSize: '0.7rem',
            height: 22,
            mt: 0.5,
            borderColor: alpha(theme.palette.primary.main, 0.3),
            color: 'text.secondary',
          }}
        />
      )}
      
      {/* Action Buttons */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1.5 }}>
        {isSolutionActive && onReset && (
          <Button
            fullWidth
            variant="contained"
            size="small"
            startIcon={<AddIcon />}
            onClick={onReset}
            sx={{
              textTransform: 'none',
              fontSize: '0.75rem',
              py: 0.75,
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || theme.palette.primary.dark})`,
              '&:hover': {
                transform: 'translateY(-1px)',
                boxShadow: theme.shadows[4],
              },
              transition: 'all 0.2s ease',
            }}
          >
            New Story
          </Button>
        )}
        <Button
          fullWidth
          variant="outlined"
          size="small"
          startIcon={<EditIcon />}
          onClick={onExpand}
          sx={{
            textTransform: 'none',
            fontSize: '0.75rem',
            py: 0.75,
            borderWidth: 1.5,
            '&:hover': {
              borderWidth: 1.5,
              transform: 'translateY(-1px)',
              boxShadow: theme.shadows[2],
            },
            transition: 'all 0.2s ease',
          }}
        >
          {isSolutionActive ? 'View Form' : 'Edit Story'}
        </Button>
      </Box>
    </Paper>
  );
};

export default CollapsedStorySummary;

