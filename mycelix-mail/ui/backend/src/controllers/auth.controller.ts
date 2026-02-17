import { Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { authService } from '../services/auth.service';
import { asyncHandler } from '../middleware/errorHandler';

export class AuthController {
  register = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { email, password, firstName, lastName } = req.body;
    const result = await authService.register(email, password, firstName, lastName);

    res.status(201).json({
      status: 'success',
      data: result,
    });
  });

  login = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { email, password } = req.body;
    const result = await authService.login(email, password);

    res.json({
      status: 'success',
      data: result,
    });
  });

  getProfile = asyncHandler(async (req: AuthRequest, res: Response) => {
    const user = await authService.getProfile(req.userId!);

    res.json({
      status: 'success',
      data: { user },
    });
  });

  updateProfile = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { firstName, lastName, email } = req.body;
    const user = await authService.updateProfile(req.userId!, { firstName, lastName, email });

    res.json({
      status: 'success',
      data: { user },
    });
  });

  changePassword = asyncHandler(async (req: AuthRequest, res: Response) => {
    const { currentPassword, newPassword } = req.body;
    const result = await authService.changePassword(req.userId!, currentPassword, newPassword);

    res.json({
      status: 'success',
      data: result,
    });
  });
}

export const authController = new AuthController();
